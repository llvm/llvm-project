import lldb

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

import os
import platform
import shutil
import time
import socket


class PlatformSDKTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)
    NO_DEBUG_INFO_TESTCASE = True

    # The port used by debugserver.
    PORT = 54637

    # The number of attempts.
    ATTEMPTS = 10

    # Time given to the binary to launch and to debugserver to attach to it for
    # every attempt. We'll wait a maximum of 10 times 2 seconds while the
    # inferior will wait 10 times 10 seconds.
    TIMEOUT = 2

    def no_debugserver(self):
        if os.getenv('LLDB_DEBUGSERVER_PATH') is None:
            return 'no debugserver'
        return None

    def port_not_available(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        if s.connect_ex(('127.0.0.1', self.PORT)) == 0:
            return '{} not available'.format(self.PORT)
        return None

    @no_debug_info_test
    @skipUnlessDarwin
    @expectedFailureIfFn(no_debugserver)
    @expectedFailureIfFn(port_not_available)
    def test_macos_sdk(self):
        self.build()

        exe = self.getBuildArtifact('a.out')
        token = self.getBuildArtifact('token')

        # Remove the old token.
        try:
            os.remove(token)
        except:
            pass

        # Create a fake 'SDK' directory.
        test_home = os.path.join(self.getBuildDir(), 'fake_home.noindex')
        macos_version = platform.mac_ver()[0]
        sdk_dir = os.path.join(test_home, 'Library', 'Developer', 'Xcode',
                               'macOS DeviceSupport', macos_version)
        symbols_dir = os.path.join(sdk_dir, 'Symbols')
        lldbutil.mkdir_p(symbols_dir)

        # Save the current home directory and restore it afterwards.
        old_home = os.getenv('HOME')

        def cleanup():
            if not old_home:
                del os.environ['HOME']
            else:
                os.environ['HOME'] = old_home

        self.addTearDownHook(cleanup)
        os.environ['HOME'] = test_home

        # Launch our test binary.
        inferior = self.spawnSubprocess(exe, [token])
        pid = inferior.pid

        # Wait for the binary to launch.
        lldbutil.wait_for_file_on_target(self, token)

        # Move the binary into the 'SDK'.
        rel_exe_path = os.path.relpath(exe, '/')
        exe_sdk_path = os.path.join(symbols_dir, rel_exe_path)
        lldbutil.mkdir_p(os.path.dirname(exe_sdk_path))
        shutil.move(exe, exe_sdk_path)

        # Attach to it with debugserver.
        debugserver = os.getenv('LLDB_DEBUGSERVER_PATH')
        debugserver_args = [
            'localhost:{}'.format(self.PORT), '--attach={}'.format(pid)
        ]
        self.spawnSubprocess(debugserver, debugserver_args)

        # Select the platform.
        self.expect('platform select remote-macosx', substrs=[sdk_dir])

        # Connect to debugserver
        interpreter = self.dbg.GetCommandInterpreter()
        connected = False
        for i in range(self.ATTEMPTS):
            result = lldb.SBCommandReturnObject()
            interpreter.HandleCommand('gdb-remote {}'.format(self.PORT),
                                      result)
            connected = result.Succeeded()
            if connected:
                break
            time.sleep(self.TIMEOUT)

        self.assertTrue(connected, "could not connect to debugserver")

        # Make sure the image was loaded from the 'SDK'.
        self.expect('image list', substrs=[exe_sdk_path])
