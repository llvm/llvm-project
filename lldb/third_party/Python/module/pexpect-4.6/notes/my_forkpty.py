import os, fcntl, termios
import time

def my_forkpty():

    (master_fd, slave_fd) = os.openpty()

    if (master_fd < 0  or  slave_fd < 0):
        raise ExceptionPexpect("Forkpty failed")

    # slave_name = ptsname(master_fd);

    pid = os.fork();
    if pid == -1:
        raise ExceptionPexpect("Forkpty failed")
    elif pid == 0: # Child
        if hasattr(termios, 'TIOCNOTTY'):
        #        Some platforms require an explicit detach of the
        #        current controlling tty before closing stdin, stdout, stderr.
        #        OpenBSD says that this is obsolete, but doesn't hurt.
            try:
                fd = os.open("/dev/tty", os.O_RDWR | os.O_NOCTTY)
            except:
                pass
            else: #if fd >= 0:
                fcntl.ioctl(fd, termios.TIOCNOTTY, 0)
                os.close(fd)

        # The setsid() system call will place the process into its own session
        # which has the effect of disassociating it from the controlling terminal.
        # This is known to be true for OpenBSD.
        os.setsid()
        # except:            return posix_error();

        # Verify that we are disconnected from the controlling tty.
        try:
            fd = os.open("/dev/tty", os.O_RDWR | os.O_NOCTTY)
            os.close(fd)
            raise ExceptionPexpect("Forkpty failed")
        except:
            pass
        if 'TIOCSCTTY' in dir(termios):
            # Make the pseudo terminal the controlling terminal for this process
            # (the process must not currently have a controlling terminal).
            if fcntl.ioctl(slave_fd, termios.TIOCSCTTY, '') < 0:
                raise ExceptionPexpect("Forkpty failed")

#        # Verify that we can open to the slave pty file. */
#        fd = os.open(slave_name, os.O_RDWR);
#        if fd < 0:
#            raise ExceptionPexpect("Forkpty failed")
#        else:
#            os.close(fd);

        # Verify that we now have a controlling tty.
        fd = os.open("/dev/tty", os.O_WRONLY)
        if fd < 0:
            raise ExceptionPexpect("This process could not get a controlling tty.")
        else:
            os.close(fd)

        os.close(master_fd)
        os.dup2(slave_fd, 0)
        os.dup2(slave_fd, 1)
        os.dup2(slave_fd, 2)
        if slave_fd > 2:
            os.close(slave_fd)
        pid = 0

    else:
        # PARENT 
        os.close(slave_fd);

    if pid == -1:
        raise ExceptionPexpect("This process could not get a controlling tty.")
#    if (pid == 0)
#        PyOS_AfterFork();

    return (pid, master_fd)

pid, fd = my_forkpty ()
if pid == 0: # child
    print 'I am not a robot!'
else:
    print '(pid, fd) = (%d, %d)' % (pid, fd)
    time.sleep(1) # Give the child a chance to print.
    print 'Robots always say:', os.read(fd,100)
    os.close(fd)

