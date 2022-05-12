#!/usr/bin/env python3


import rospy

from std_msgs.msg import Float64
from std_msgs.msg import String

class speedNODE():
    def __init__(self):
        """It forwards the control messages received from socket to the serial handling node. 
        """
        
        rospy.init_node('speedNODE', anonymous=False)

        # Command publisher object
        self.command_publisher = rospy.Publisher("/automobile/command", String, queue_size=1)

    def speed_callback(self, data):
        #rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data)
        #take value from data and use it as a paramter to make it into a command and publish
        #{'action': '2', 'steerAngle': 0.0}

        command = "{'action': '1', 'speed': " + str(data.data) + "}"
        command = command.replace("'", '"') #must replace '' for json formate (this was easier than regex)

        print(command)
        self.command_publisher.publish(command) #send command to serialNODE


     # ===================================== RUN ==========================================
    def run(self):
        """Apply the initializing methods and start the threads
        """
        rospy.loginfo("starting speedNODE")
        self._read_correction()

    def _read_correction(self):
        """Receive the message and forwards them to the Serial Handler Node. 
        """

        while not rospy.is_shutdown():

            rospy.init_node('speedNODE', anonymous=False)
            command = self.command_subscriber = rospy.Subscriber("SpeedAdjustment", Float64, self.speed_callback) #read command from lane centering 
            #print(command)
            #convert command string to right formate and publish string 
            #self.command_publisher.publish(command)
    

            
if __name__ == "__main__":
    speedNod = speedNODE()
    speedNod.run()
