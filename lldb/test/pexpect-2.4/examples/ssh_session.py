#
# Eric S. Raymond
#
# Greatly modified by Nigel W. Moriarty
# April 2003
#
from pexpect import *
import os, sys
import getpass
import time
    
class ssh_session:

    "Session with extra state including the password to be used."

    def __init__(self, user, host, password=None, verbose=0):

        self.user = user
        self.host = host
        self.verbose = verbose
        self.password = password
        self.keys = [
            'authenticity',
            'assword:',
            '@@@@@@@@@@@@',
            'Command not found.',
            EOF,
            ]
        
        self.f = open('ssh.out','w')
            
    def __repr__(self):

        outl = 'class :'+self.__class__.__name__
        for attr in self.__dict__:
            if attr == 'password':
                outl += '\n\t'+attr+' : '+'*'*len(self.password)
            else:
                outl += '\n\t'+attr+' : '+str(getattr(self, attr))
        return outl

    def __exec(self, command):

        "Execute a command on the remote host.    Return the output."
        child = spawn(command,
                                    #timeout=10,
                                    )
        if self.verbose:
            sys.stderr.write("-> " + command + "\n")
        seen = child.expect(self.keys)
        self.f.write(str(child.before) + str(child.after)+'\n')
        if seen == 0:
            child.sendline('yes')
            seen = child.expect(self.keys)
        if seen == 1:
            if not self.password:
                self.password = getpass.getpass('Remote password: ')
            child.sendline(self.password)
            child.readline()
            time.sleep(5)
            # Added to allow the background running of remote process
            if not child.isalive():
                seen = child.expect(self.keys)
        if seen == 2: 
            lines = child.readlines()
            self.f.write(lines)
        if self.verbose:
            sys.stderr.write("<- " + child.before + "|\n")
        try:
            self.f.write(str(child.before) + str(child.after)+'\n')
        except:
            pass
        self.f.close()
        return child.before

    def ssh(self, command):

        return self.__exec("ssh -l %s %s \"%s\"" \
                                             % (self.user,self.host,command))

    def scp(self, src, dst):

        return self.__exec("scp %s %s@%s:%s" \
                                             % (src, session.user, session.host, dst))

    def exists(self, file):

        "Retrieve file permissions of specified remote file."
        seen = self.ssh("/bin/ls -ld %s" % file)
        if string.find(seen, "No such file") > -1:
            return None # File doesn't exist
        else:
            return seen.split()[0] # Return permission field of listing.

