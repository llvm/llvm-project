"""This class extends pexpect.spawn to specialize setting up SSH connections.
This adds methods for login, logout, and expecting the shell prompt.

$Id: pxssh.py 513 2008-02-09 18:26:13Z noah $
"""

from pexpect import *
import pexpect
import time

__all__ = ['ExceptionPxssh', 'pxssh']

# Exception classes used by this module.


class ExceptionPxssh(ExceptionPexpect):
    """Raised for pxssh exceptions.
    """


class pxssh (spawn):

    """This class extends pexpect.spawn to specialize setting up SSH
    connections. This adds methods for login, logout, and expecting the shell
    prompt. It does various tricky things to handle many situations in the SSH
    login process. For example, if the session is your first login, then pxssh
    automatically accepts the remote certificate; or if you have public key
    authentication setup then pxssh won't wait for the password prompt.

    pxssh uses the shell prompt to synchronize output from the remote host. In
    order to make this more robust it sets the shell prompt to something more
    unique than just $ or #. This should work on most Borne/Bash or Csh style
    shells.

    Example that runs a few commands on a remote server and prints the result::

        import pxssh
        import getpass
        try:
            s = pxssh.pxssh()
            hostname = raw_input('hostname: ')
            username = raw_input('username: ')
            password = getpass.getpass('password: ')
            s.login (hostname, username, password)
            s.sendline ('uptime')  # run a command
            s.prompt()             # match the prompt
            print s.before         # print everything before the prompt.
            s.sendline ('ls -l')
            s.prompt()
            print s.before
            s.sendline ('df')
            s.prompt()
            print s.before
            s.logout()
        except pxssh.ExceptionPxssh, e:
            print "pxssh failed on login."
            print str(e)

    Note that if you have ssh-agent running while doing development with pxssh
    then this can lead to a lot of confusion. Many X display managers (xdm,
    gdm, kdm, etc.) will automatically start a GUI agent. You may see a GUI
    dialog box popup asking for a password during development. You should turn
    off any key agents during testing. The 'force_password' attribute will turn
    off public key authentication. This will only work if the remote SSH server
    is configured to allow password logins. Example of using 'force_password'
    attribute::

            s = pxssh.pxssh()
            s.force_password = True
            hostname = raw_input('hostname: ')
            username = raw_input('username: ')
            password = getpass.getpass('password: ')
            s.login (hostname, username, password)
    """

    def __init__(
            self,
            timeout=30,
            maxread=2000,
            searchwindowsize=None,
            logfile=None,
            cwd=None,
            env=None):
        spawn.__init__(
            self,
            None,
            timeout=timeout,
            maxread=maxread,
            searchwindowsize=searchwindowsize,
            logfile=logfile,
            cwd=cwd,
            env=env)

        self.name = '<pxssh>'

        # SUBTLE HACK ALERT! Note that the command to set the prompt uses a
        # slightly different string than the regular expression to match it. This
        # is because when you set the prompt the command will echo back, but we
        # don't want to match the echoed command. So if we make the set command
        # slightly different than the regex we eliminate the problem. To make the
        # set command different we add a backslash in front of $. The $ doesn't
        # need to be escaped, but it doesn't hurt and serves to make the set
        # prompt command different than the regex.

        # used to match the command-line prompt
        self.UNIQUE_PROMPT = "\[PEXPECT\][\$\#] "
        self.PROMPT = self.UNIQUE_PROMPT

        # used to set shell command-line prompt to UNIQUE_PROMPT.
        self.PROMPT_SET_SH = "PS1='[PEXPECT]\$ '"
        self.PROMPT_SET_CSH = "set prompt='[PEXPECT]\$ '"
        self.SSH_OPTS = "-o'RSAAuthentication=no' -o 'PubkeyAuthentication=no'"
        # Disabling X11 forwarding gets rid of the annoying SSH_ASKPASS from
        # displaying a GUI password dialog. I have not figured out how to
        # disable only SSH_ASKPASS without also disabling X11 forwarding.
        # Unsetting SSH_ASKPASS on the remote side doesn't disable it! Annoying!
        #self.SSH_OPTS = "-x -o'RSAAuthentication=no' -o 'PubkeyAuthentication=no'"
        self.force_password = False
        self.auto_prompt_reset = True

    def levenshtein_distance(self, a, b):
        """This calculates the Levenshtein distance between a and b.
        """

        n, m = len(a), len(b)
        if n > m:
            a, b = b, a
            n, m = m, n
        current = range(n + 1)
        for i in range(1, m + 1):
            previous, current = current, [i] + [0] * n
            for j in range(1, n + 1):
                add, delete = previous[j] + 1, current[j - 1] + 1
                change = previous[j - 1]
                if a[j - 1] != b[i - 1]:
                    change = change + 1
                current[j] = min(add, delete, change)
        return current[n]

    def sync_original_prompt(self):
        """This attempts to find the prompt. Basically, press enter and record
        the response; press enter again and record the response; if the two
        responses are similar then assume we are at the original prompt. This
        is a slow function. It can take over 10 seconds. """

        # All of these timing pace values are magic.
        # I came up with these based on what seemed reliable for
        # connecting to a heavily loaded machine I have.
        # If latency is worse than these values then this will fail.

        try:
            # GAS: Clear out the cache before getting the prompt
            self.read_nonblocking(size=10000, timeout=1)
        except TIMEOUT:
            pass
        time.sleep(0.1)
        self.sendline()
        time.sleep(0.5)
        x = self.read_nonblocking(size=1000, timeout=1)
        time.sleep(0.1)
        self.sendline()
        time.sleep(0.5)
        a = self.read_nonblocking(size=1000, timeout=1)
        time.sleep(0.1)
        self.sendline()
        time.sleep(0.5)
        b = self.read_nonblocking(size=1000, timeout=1)
        ld = self.levenshtein_distance(a, b)
        len_a = len(a)
        if len_a == 0:
            return False
        if float(ld) / len_a < 0.4:
            return True
        return False

    # TODO: This is getting messy and I'm pretty sure this isn't perfect.
    # TODO: I need to draw a flow chart for this.
    def login(
            self,
            server,
            username,
            password='',
            terminal_type='ansi',
            original_prompt=r"[#$]",
            login_timeout=10,
            port=None,
            auto_prompt_reset=True):
        """This logs the user into the given server. It uses the
        'original_prompt' to try to find the prompt right after login. When it
        finds the prompt it immediately tries to reset the prompt to something
        more easily matched. The default 'original_prompt' is very optimistic
        and is easily fooled. It's more reliable to try to match the original
        prompt as exactly as possible to prevent false matches by server
        strings such as the "Message Of The Day". On many systems you can
        disable the MOTD on the remote server by creating a zero-length file
        called "~/.hushlogin" on the remote server. If a prompt cannot be found
        then this will not necessarily cause the login to fail. In the case of
        a timeout when looking for the prompt we assume that the original
        prompt was so weird that we could not match it, so we use a few tricks
        to guess when we have reached the prompt. Then we hope for the best and
        blindly try to reset the prompt to something more unique. If that fails
        then login() raises an ExceptionPxssh exception.

        In some situations it is not possible or desirable to reset the
        original prompt. In this case, set 'auto_prompt_reset' to False to
        inhibit setting the prompt to the UNIQUE_PROMPT. Remember that pxssh
        uses a unique prompt in the prompt() method. If the original prompt is
        not reset then this will disable the prompt() method unless you
        manually set the PROMPT attribute. """

        ssh_options = '-q'
        if self.force_password:
            ssh_options = ssh_options + ' ' + self.SSH_OPTS
        if port is not None:
            ssh_options = ssh_options + ' -p %s' % (str(port))
        cmd = "ssh %s -l %s %s" % (ssh_options, username, server)

        # This does not distinguish between a remote server 'password' prompt
        # and a local ssh 'passphrase' prompt (for unlocking a private key).
        spawn._spawn(self, cmd)
        i = self.expect(
            [
                "(?i)are you sure you want to continue connecting",
                original_prompt,
                "(?i)(?:password)|(?:passphrase for key)",
                "(?i)permission denied",
                "(?i)terminal type",
                TIMEOUT,
                "(?i)connection closed by remote host"],
            timeout=login_timeout)

        # First phase
        if i == 0:
            # New certificate -- always accept it.
            # This is what you get if SSH does not have the remote host's
            # public key stored in the 'known_hosts' cache.
            self.sendline("yes")
            i = self.expect(
                [
                    "(?i)are you sure you want to continue connecting",
                    original_prompt,
                    "(?i)(?:password)|(?:passphrase for key)",
                    "(?i)permission denied",
                    "(?i)terminal type",
                    TIMEOUT])
        if i == 2:  # password or passphrase
            self.sendline(password)
            i = self.expect(
                [
                    "(?i)are you sure you want to continue connecting",
                    original_prompt,
                    "(?i)(?:password)|(?:passphrase for key)",
                    "(?i)permission denied",
                    "(?i)terminal type",
                    TIMEOUT])
        if i == 4:
            self.sendline(terminal_type)
            i = self.expect(
                [
                    "(?i)are you sure you want to continue connecting",
                    original_prompt,
                    "(?i)(?:password)|(?:passphrase for key)",
                    "(?i)permission denied",
                    "(?i)terminal type",
                    TIMEOUT])

        # Second phase
        if i == 0:
            # This is weird. This should not happen twice in a row.
            self.close()
            raise ExceptionPxssh(
                'Weird error. Got "are you sure" prompt twice.')
        elif i == 1:  # can occur if you have a public key pair set to authenticate.
            # TODO: May NOT be OK if expect() got tricked and matched a false
            # prompt.
            pass
        elif i == 2:  # password prompt again
            # For incorrect passwords, some ssh servers will
            # ask for the password again, others return 'denied' right away.
            # If we get the password prompt again then this means
            # we didn't get the password right the first time.
            self.close()
            raise ExceptionPxssh('password refused')
        elif i == 3:  # permission denied -- password was bad.
            self.close()
            raise ExceptionPxssh('permission denied')
        elif i == 4:  # terminal type again? WTF?
            self.close()
            raise ExceptionPxssh(
                'Weird error. Got "terminal type" prompt twice.')
        elif i == 5:  # Timeout
            # This is tricky... I presume that we are at the command-line prompt.
            # It may be that the shell prompt was so weird that we couldn't match
            # it. Or it may be that we couldn't log in for some other reason. I
            # can't be sure, but it's safe to guess that we did login because if
            # I presume wrong and we are not logged in then this should be caught
            # later when I try to set the shell prompt.
            pass
        elif i == 6:  # Connection closed by remote host
            self.close()
            raise ExceptionPxssh('connection closed')
        else:  # Unexpected
            self.close()
            raise ExceptionPxssh('unexpected login response')
        if not self.sync_original_prompt():
            self.close()
            raise ExceptionPxssh('could not synchronize with original prompt')
        # We appear to be in.
        # set shell prompt to something unique.
        if auto_prompt_reset:
            if not self.set_unique_prompt():
                self.close()
                raise ExceptionPxssh(
                    'could not set shell prompt\n' + self.before)
        return True

    def logout(self):
        """This sends exit to the remote shell. If there are stopped jobs then
        this automatically sends exit twice. """

        self.sendline("exit")
        index = self.expect([EOF, "(?i)there are stopped jobs"])
        if index == 1:
            self.sendline("exit")
            self.expect(EOF)
        self.close()

    def prompt(self, timeout=20):
        """This matches the shell prompt. This is little more than a short-cut
        to the expect() method. This returns True if the shell prompt was
        matched. This returns False if there was a timeout. Note that if you
        called login() with auto_prompt_reset set to False then you should have
        manually set the PROMPT attribute to a regex pattern for matching the
        prompt. """

        i = self.expect([self.PROMPT, TIMEOUT], timeout=timeout)
        if i == 1:
            return False
        return True

    def set_unique_prompt(self):
        """This sets the remote prompt to something more unique than # or $.
        This makes it easier for the prompt() method to match the shell prompt
        unambiguously. This method is called automatically by the login()
        method, but you may want to call it manually if you somehow reset the
        shell prompt. For example, if you 'su' to a different user then you
        will need to manually reset the prompt. This sends shell commands to
        the remote host to set the prompt, so this assumes the remote host is
        ready to receive commands.

        Alternatively, you may use your own prompt pattern. Just set the PROMPT
        attribute to a regular expression that matches it. In this case you
        should call login() with auto_prompt_reset=False; then set the PROMPT
        attribute. After that the prompt() method will try to match your prompt
        pattern."""

        self.sendline("unset PROMPT_COMMAND")
        self.sendline(self.PROMPT_SET_SH)  # sh-style
        i = self.expect([TIMEOUT, self.PROMPT], timeout=10)
        if i == 0:  # csh-style
            self.sendline(self.PROMPT_SET_CSH)
            i = self.expect([TIMEOUT, self.PROMPT], timeout=10)
            if i == 0:
                return False
        return True

# vi:ts=4:sw=4:expandtab:ft=python:
