Examples
========

Under the distribution tarball directory you should find an "examples" directory.
This is the best way to learn to use Pexpect. See the descriptions of Pexpect
Examples.

`topip.py <https://github.com/pexpect/pexpect/blob/master/examples/topip.py>`_
  This runs `netstat` on a local or remote server. It calculates some simple
  statistical information on the number of external inet connections. This can
  be used to detect if one IP address is taking up an excessive number of
  connections. It can also send an email alert if a given IP address exceeds a
  threshold between runs of the script. This script can be used as a drop-in
  Munin plugin or it can be used stand-alone from cron. I used this on a busy
  web server that would sometimes get hit with denial of service attacks. This
  made it easy to see if a script was opening many multiple connections. A
  typical browser would open fewer than 10 connections at once. A script might
  open over 100 simultaneous connections.

`hive.py <https://github.com/pexpect/pexpect/blob/master/examples/hive.py>`_
  This script creates SSH connections to a list of hosts that you provide.
  Then you are given a command line prompt. Each shell command that you
  enter is sent to all the hosts. The response from each host is collected
  and printed. For example, you could connect to a dozen different
  machines and reboot them all at once.

`script.py <https://github.com/pexpect/pexpect/blob/master/examples/script.py>`_
  This implements a command similar to the classic BSD "script" command.
  This will start a subshell and log all input and output to a file.
  This demonstrates the :meth:`~pexpect.spawn.interact` method of Pexpect.

`ftp.py <https://github.com/pexpect/pexpect/blob/master/examples/ftp.py>`_
  This demonstrates an FTP "bookmark". This connects to an ftp site;
  does a few ftp tasks; and then gives the user interactive control over
  the session. In this case the "bookmark" is to a directory on the
  OpenBSD ftp server. It puts you in the i386 packages directory. You
  can easily modify this for other sites. This demonstrates the
  :meth:`~pexpect.spawn.interact` method of Pexpect.

`monitor.py <https://github.com/pexpect/pexpect/blob/master/examples/monitor.py>`_
  This runs a sequence of commands on a remote host using SSH. It runs a
  simple system checks such as uptime and free to monitor the state of
  the remote host.

`passmass.py <https://github.com/pexpect/pexpect/blob/master/examples/passmass.py>`_
  This will login to each given server and change the password of the
  given user. This demonstrates scripting logins and passwords.

`python.py <https://github.com/pexpect/pexpect/blob/master/examples/python.py>`_
  This starts the python interpreter and prints the greeting message
  backwards. It then gives the user iteractive control of Python. It's
  pretty useless!

`ssh_tunnel.py <https://github.com/pexpect/pexpect/blob/master/examples/ssh_tunnel.py>`_
  This starts an SSH tunnel to a remote machine. It monitors the
  connection and restarts the tunnel if it goes down.

`uptime.py <https://github.com/pexpect/pexpect/blob/master/examples/uptime.py>`_
  This will run the uptime command and parse the output into variables.
  This demonstrates using a single regular expression to match the
  output of a command and capturing different variable in match groups.
  The grouping regular expression handles a wide variety of different
  uptime formats. 
