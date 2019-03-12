import os

filename = os.tmpnam()
print 'filename:', filename

fd_out = os.open(filename, os.O_CREAT | os.O_WRONLY)
print 'fd_out:', fd_out
os.write (fd_out, 'This is a test.\n')
os.close(fd_out)
print
print 'testing read on good fd...'
fd_in = os.open (filename, os.O_RDONLY)
print 'fd_in:', fd_in
while 1:
	data_in = os.read(fd_in, 1)
	print 'data_in:', data_in
	if data_in == '':
		print 'data_in was empty'
		break #sys.exit(1)
os.close(fd_in)
print
print
print 'testing read on closed fd...'
fd_in = os.open ('test_read.py', os.O_RDONLY)
print 'fd_in:', fd_in
while 1:
	data_in = os.read(fd_in, 1)
	print 'data_in:', data_in
	if data_in == '':
		print 'data_in was empty'
		break
os.close(fd_in)
d = os.read(fd_in, 1) # fd_in should be closed now...
if s == '':
	print 'd is empty. good.'
