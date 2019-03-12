#!/usr/bin/env python
import os
import tempfile
import unittest

from pexpect import pxssh

class SSHTestBase(unittest.TestCase):
    def setUp(self):
        self.orig_path = os.environ.get('PATH')
        fakessh_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'fakessh'))
        os.environ['PATH'] = fakessh_dir + \
                    ((os.pathsep + self.orig_path) if self.orig_path else '')

    def tearDown(self):
        if self.orig_path:
            os.environ['PATH'] = self.orig_path
        else:
            del os.environ['PATH']

class PxsshTestCase(SSHTestBase):
    def test_fake_ssh(self):
        ssh = pxssh.pxssh()
        #ssh.logfile_read = sys.stdout  # DEBUG
        ssh.login('server', 'me', password='s3cret')
        ssh.sendline('ping')
        ssh.expect('pong', timeout=10)
        assert ssh.prompt(timeout=10)
        ssh.logout()

    def test_wrong_pw(self):
        ssh = pxssh.pxssh()
        try:
            ssh.login('server', 'me', password='wr0ng')
        except pxssh.ExceptionPxssh:
            pass
        else:
            assert False, 'Password should have been refused'

    def test_failed_set_unique_prompt(self):
        ssh = pxssh.pxssh()
        ssh.set_unique_prompt = lambda: False
        try:
            ssh.login('server', 'me', password='s3cret',
                      auto_prompt_reset=True)
        except pxssh.ExceptionPxssh:
            pass
        else:
            assert False, 'should have raised exception, pxssh.ExceptionPxssh'

    def test_connection_refused(self):
        ssh = pxssh.pxssh()
        try:
            ssh.login('noserver', 'me', password='s3cret')
        except pxssh.ExceptionPxssh:
            pass
        else:
            assert False, 'should have raised exception, pxssh.ExceptionPxssh'

    def test_ssh_tunnel_string(self):
        ssh = pxssh.pxssh(debug_command_string=True)
        tunnels = { 'local': ['2424:localhost:22'],'remote': ['2525:localhost:22'],
            'dynamic': [8888] }
        confirmation_strings = 0
        confirmation_array = ['-R 2525:localhost:22','-L 2424:localhost:22','-D 8888']
        string = ssh.login('server', 'me', password='s3cret', ssh_tunnels=tunnels)
        for confirmation in confirmation_array:
            if confirmation in string:
                confirmation_strings+=1

        if confirmation_strings!=len(confirmation_array):
            assert False, 'String generated from tunneling is incorrect.'

    def test_remote_ssh_tunnel_string(self):
        ssh = pxssh.pxssh(debug_command_string=True)
        tunnels = { 'local': ['2424:localhost:22'],'remote': ['2525:localhost:22'],
            'dynamic': [8888] }
        confirmation_strings = 0
        confirmation_array = ['-R 2525:localhost:22','-L 2424:localhost:22','-D 8888']
        string = ssh.login('server', 'me', password='s3cret', ssh_tunnels=tunnels, spawn_local_ssh=False)
        for confirmation in confirmation_array:
            if confirmation in string:
                confirmation_strings+=1

        if confirmation_strings!=len(confirmation_array):
            assert False, 'String generated from remote tunneling is incorrect.'

    def test_ssh_config_passing_string(self):
        ssh = pxssh.pxssh(debug_command_string=True)
        (temp_file,config_path) = tempfile.mkstemp()
        string = ssh.login('server', 'me', password='s3cret', spawn_local_ssh=False, ssh_config=config_path)
        if not '-F '+config_path in string:
            assert False, 'String generated from SSH config passing is incorrect.'

    def test_ssh_key_string(self):
        ssh = pxssh.pxssh(debug_command_string=True)
        confirmation_strings = 0
        confirmation_array = [' -A']
        string = ssh.login('server', 'me', password='s3cret', ssh_key=True)
        for confirmation in confirmation_array:
            if confirmation in string:
                confirmation_strings+=1

        if confirmation_strings!=len(confirmation_array):
            assert False, 'String generated from forcing the SSH agent sock is incorrect.'

        confirmation_strings = 0
        (temp_file,ssh_key) = tempfile.mkstemp()
        confirmation_array = [' -i '+ssh_key]
        string = ssh.login('server', 'me', password='s3cret', ssh_key=ssh_key)
        for confirmation in confirmation_array:
            if confirmation in string:
                confirmation_strings+=1
        
        if confirmation_strings!=len(confirmation_array):
            assert False, 'String generated from adding an SSH key is incorrect.'


if __name__ == '__main__':
    unittest.main()
