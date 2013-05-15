#!/usr/bin/env python 

""" 
A simple echo server.
Taken from http://docs.python.org/library/socket.html#example.
"""

import socket

HOST = 'localhost'        # Symbolic name meaning local interfaces
PORT = 12345              # Arbitrary non-privileged port
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((HOST, PORT))
print '\nListening on %s:%d' % (HOST, PORT)
s.listen(1)
conn, addr = s.accept()
print 'Connected by', addr
while 1:
    data = conn.recv(1024)
    if not data: break
    conn.send(data)
    print 'Received:', data
conn.close()
