#!/usr/bin/env python

"""Back door shell server

This exposes an shell terminal on a socket.

    --hostname : sets the remote host name to open an ssh connection to.
    --username : sets the user name to login with
    --password : (optional) sets the password to login with
    --port     : set the local port for the server to listen on
    --watch    : show the virtual screen after each client request
"""

# Having the password on the command line is not a good idea, but
# then this entire project is probably not the most security concious thing
# I've ever built. This should be considered an experimental tool -- at best.
import pxssh, pexpect, ANSI
import time, sys, os, getopt, getpass, traceback, threading, socket

def exit_with_usage(exit_code=1):

    print globals()['__doc__']
    os._exit(exit_code)

class roller (threading.Thread):

    """This runs a function in a loop in a thread."""

    def __init__(self, interval, function, args=[], kwargs={}):

        """The interval parameter defines time between each call to the function.
        """

        threading.Thread.__init__(self)
        self.interval = interval
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self.finished = threading.Event()

    def cancel(self):

        """Stop the roller."""

        self.finished.set()

    def run(self):

        while not self.finished.isSet():
            # self.finished.wait(self.interval)
            self.function(*self.args, **self.kwargs)

def endless_poll (child, prompt, screen, refresh_timeout=0.1):

    """This keeps the screen updated with the output of the child. This runs in
    a separate thread. See roller(). """

    #child.logfile_read = screen
    try:
        s = child.read_nonblocking(4000, 0.1)
        screen.write(s)
    except:
        pass
    #while True:
    #    #child.prompt (timeout=refresh_timeout)
    #    try:
    #        #child.read_nonblocking(1,timeout=refresh_timeout)
    #        child.read_nonblocking(4000, 0.1)
    #    except:
    #        pass

def daemonize (stdin='/dev/null', stdout='/dev/null', stderr='/dev/null'):

    '''This forks the current process into a daemon. Almost none of this is
    necessary (or advisable) if your daemon is being started by inetd. In that
    case, stdin, stdout and stderr are all set up for you to refer to the
    network connection, and the fork()s and session manipulation should not be
    done (to avoid confusing inetd). Only the chdir() and umask() steps remain
    as useful. 

    References:
        UNIX Programming FAQ
        1.7 How do I get my program to act like a daemon?
        http://www.erlenstar.demon.co.uk/unix/faq_2.html#SEC16

        Advanced Programming in the Unix Environment
        W. Richard Stevens, 1992, Addison-Wesley, ISBN 0-201-56317-7.

    The stdin, stdout, and stderr arguments are file names that will be opened
    and be used to replace the standard file descriptors in sys.stdin,
    sys.stdout, and sys.stderr. These arguments are optional and default to
    /dev/null. Note that stderr is opened unbuffered, so if it shares a file
    with stdout then interleaved output may not appear in the order that you
    expect. '''

    # Do first fork.
    try: 
        pid = os.fork() 
        if pid > 0:
            sys.exit(0)   # Exit first parent.
    except OSError, e: 
        sys.stderr.write ("fork #1 failed: (%d) %s\n" % (e.errno, e.strerror) )
        sys.exit(1)

    # Decouple from parent environment.
    os.chdir("/") 
    os.umask(0) 
    os.setsid() 

    # Do second fork.
    try: 
        pid = os.fork() 
        if pid > 0:
            sys.exit(0)   # Exit second parent.
    except OSError, e: 
        sys.stderr.write ("fork #2 failed: (%d) %s\n" % (e.errno, e.strerror) )
        sys.exit(1)

    # Now I am a daemon!
    
    # Redirect standard file descriptors.
    si = open(stdin, 'r')
    so = open(stdout, 'a+')
    se = open(stderr, 'a+', 0)
    os.dup2(si.fileno(), sys.stdin.fileno())
    os.dup2(so.fileno(), sys.stdout.fileno())
    os.dup2(se.fileno(), sys.stderr.fileno())

    # I now return as the daemon
    return 0

def add_cursor_blink (response, row, col):

    i = (row-1) * 80 + col
    return response[:i]+'<img src="http://www.noah.org/cursor.gif">'+response[i:]

def main ():

    try:
        optlist, args = getopt.getopt(sys.argv[1:], 'h?d', ['help','h','?', 'hostname=', 'username=', 'password=', 'port=', 'watch'])
    except Exception, e:
        print str(e)
        exit_with_usage()

    command_line_options = dict(optlist)
    options = dict(optlist)
    # There are a million ways to cry for help. These are but a few of them.
    if [elem for elem in command_line_options if elem in ['-h','--h','-?','--?','--help']]:
        exit_with_usage(0)
  
    hostname = "127.0.0.1"
    port = 1664
    username = os.getenv('USER')
    password = ""
    daemon_mode = False
    if '-d' in options:
        daemon_mode = True
    if '--watch' in options:
        watch_mode = True
    else:
        watch_mode = False
    if '--hostname' in options:
        hostname = options['--hostname']
    if '--port' in options:
        port = int(options['--port'])
    if '--username' in options:
        username = options['--username']
    print "Login for %s@%s:%s" % (username, hostname, port)
    if '--password' in options:
        password = options['--password']
    else:
        password = getpass.getpass('password: ')
   
    if daemon_mode: 
        print "daemonizing server"
        daemonize()
        #daemonize('/dev/null','/tmp/daemon.log','/tmp/daemon.log')
    
    sys.stdout.write ('server started with pid %d\n' % os.getpid() )

    virtual_screen = ANSI.ANSI (24,80) 
    child = pxssh.pxssh()
    child.login (hostname, username, password)
    print 'created shell. command line prompt is', child.PROMPT
    #child.sendline ('stty -echo')
    #child.setecho(False)
    virtual_screen.write (child.before)
    virtual_screen.write (child.after)

    if os.path.exists("/tmp/mysock"): os.remove("/tmp/mysock")
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    localhost = '127.0.0.1'
    s.bind('/tmp/mysock')
    os.chmod('/tmp/mysock',0777)
    print 'Listen'
    s.listen(1)
    print 'Accept'
    #s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    #localhost = '127.0.0.1'
    #s.bind((localhost, port))
    #print 'Listen'
    #s.listen(1)

    r = roller (0.01, endless_poll, (child, child.PROMPT, virtual_screen))
    r.start()
    print "screen poll updater started in background thread"
    sys.stdout.flush()

    try:
        while True:
            conn, addr = s.accept()
            print 'Connected by', addr
            data = conn.recv(1024)
            if data[0]!=':':
                cmd = ':sendline'
                arg = data.strip()
            else:
                request = data.split(' ', 1)
                if len(request)>1:
                    cmd = request[0].strip()
                    arg = request[1].strip()
                else:
                    cmd = request[0].strip()
            if cmd == ':exit':
                r.cancel()
                break
            elif cmd == ':sendline':
                child.sendline (arg)
                #child.prompt(timeout=2)
                time.sleep(0.2)
                shell_window = str(virtual_screen)
            elif cmd == ':send' or cmd==':xsend':
                if cmd==':xsend':
                    arg = arg.decode("hex")
                child.send (arg)
                time.sleep(0.2)
                shell_window = str(virtual_screen)
            elif cmd == ':cursor':
                shell_window = '%x%x' % (virtual_screen.cur_r, virtual_screen.cur_c)
            elif cmd == ':refresh':
                shell_window = str(virtual_screen)

            response = []
            response.append (shell_window)
            #response = add_cursor_blink (response, row, col)
            sent = conn.send('\n'.join(response))
            if watch_mode: print '\n'.join(response)
            if sent < len (response):
                print "Sent is too short. Some data was cut off."
            conn.close()
    finally:
        r.cancel()
        print "cleaning up socket"
        s.close()
        if os.path.exists("/tmp/mysock"): os.remove("/tmp/mysock")
        print "done!"

def pretty_box (rows, cols, s):

    """This puts an ASCII text box around the given string, s.
    """

    top_bot = '+' + '-'*cols + '+\n'
    return top_bot + '\n'.join(['|'+line+'|' for line in s.split('\n')]) + '\n' + top_bot
    
def error_response (msg):

    response = []
    response.append ("""All commands start with :
:{REQUEST} {ARGUMENT}
{REQUEST} may be one of the following:
    :sendline: Run the ARGUMENT followed by a line feed.
    :send    : send the characters in the ARGUMENT without a line feed.
    :refresh : Use to catch up the screen with the shell if state gets out of sync.
Example:
    :sendline ls -l
You may also leave off :command and it will be assumed.
Example:
    ls -l
is equivalent to:
    :sendline ls -l
""")
    response.append (msg)
    return '\n'.join(response)

def parse_host_connect_string (hcs):

    """This parses a host connection string in the form
    username:password@hostname:port. All fields are options expcet hostname. A
    dictionary is returned with all four keys. Keys that were not included are
    set to empty strings ''. Note that if your password has the '@' character
    then you must backslash escape it. """

    if '@' in hcs:
        p = re.compile (r'(?P<username>[^@:]*)(:?)(?P<password>.*)(?!\\)@(?P<hostname>[^:]*):?(?P<port>[0-9]*)')
    else:
        p = re.compile (r'(?P<username>)(?P<password>)(?P<hostname>[^:]*):?(?P<port>[0-9]*)')
    m = p.search (hcs)
    d = m.groupdict()
    d['password'] = d['password'].replace('\\@','@')
    return d
     
if __name__ == "__main__":

    try:
        start_time = time.time()
        print time.asctime()
        main()
        print time.asctime()
        print "TOTAL TIME IN MINUTES:",
        print (time.time() - start_time) / 60.0
    except Exception, e:
        print str(e)
        tb_dump = traceback.format_exc()
        print str(tb_dump)

