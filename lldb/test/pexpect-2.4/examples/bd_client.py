#!/usr/bin/env python

"""This is a very simple client for the backdoor daemon. This is intended more
for testing rather than normal use. See bd_serv.py """

import socket
import sys, time, select

def recv_wrapper(s):
    r,w,e = select.select([s.fileno()],[],[], 2)
    if not r:
        return ''
    #cols = int(s.recv(4))
    #rows = int(s.recv(4))
    cols = 80
    rows = 24
    packet_size = cols * rows * 2 # double it for good measure
    return s.recv(packet_size)

#HOST = '' #'localhost'    # The remote host
#PORT = 1664 # The same port as used by the server
s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
s.connect(sys.argv[1])#(HOST, PORT))
time.sleep(1)
#s.setblocking(0)
#s.send('COMMAND' + '\x01' + sys.argv[1])
s.send(':sendline ' + sys.argv[2])
print recv_wrapper(s)
s.close()
sys.exit()
#while True:
#    data = recv_wrapper(s)
#    if data == '':
#        break
#    sys.stdout.write (data)
#    sys.stdout.flush()
#s.close()

