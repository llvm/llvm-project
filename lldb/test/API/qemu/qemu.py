from textwrap import dedent
import argparse
import socket
import json

import use_lldb_suite
from lldbsuite.test.gdbclientutils import *

class MyResponder(MockGDBServerResponder):
    def cont(self):
        return "W47"

class FakeEmulator(MockGDBServer):
    def __init__(self, addr):
        super().__init__(UnixServerSocket(addr))
        self.responder = MyResponder()

def main():
    parser = argparse.ArgumentParser(description=dedent("""\
            Implements a fake qemu for testing purposes. The executable program
            is not actually run. Instead a very basic mock process is presented
            to lldb. The emulated program must accept at least one argument.
            This should be a path where the emulator will dump its state. This
            allows us to check the invocation parameters.
            """))
    parser.add_argument('-g', metavar="unix-socket", required=True)
    parser.add_argument('program', help="The program to 'emulate'.")
    parser.add_argument('state_file', help="Where to dump the emulator state.")
    parsed, rest = parser.parse_known_args()
    with open(parsed.state_file, "w") as f:
        json.dump({"program":parsed.program, "rest":rest}, f)

    emulator = FakeEmulator(parsed.g)
    emulator.run()

if __name__ == "__main__":
    main()
