#!/usr/bin/env python

import json
import os
import re
import subprocess
import sys


class CrashLogPatcher:

    SYMBOL_REGEX = re.compile(r'^([0-9a-fA-F]+) T _(.*)$')
    UUID_REGEX = re.compile(r'UUID: ([-0-9a-fA-F]+) \(([^\(]+)\) .*')

    def __init__(self, data, binary, offsets):
        self.data = data
        self.binary = binary
        self.offsets = offsets

    def patch_executable(self):
        self.data = self.data.replace("@EXEC@", self.binary)
        self.data = self.data.replace("@NAME@", os.path.basename(self.binary))

    def patch_uuid(self):
        output = subprocess.check_output(['dwarfdump', '--uuid', self.binary])
        m = self.UUID_REGEX.match(output)
        if m:
            self.data = self.data.replace("@UUID@", m.group(1))

    def patch_addresses(self):
        if not self.offsets:
            return
        output = subprocess.check_output(['nm', self.binary])
        for line in output.splitlines():
            m = self.SYMBOL_REGEX.match(line)
            if m:
                address = m.group(1)
                symbol = m.group(2)
                if symbol in self.offsets:
                    patch_addr = int(m.group(1), 16) + int(
                        self.offsets[symbol])
                    self.data = self.data.replace("@{}@".format(symbol),
                                                  str(hex(patch_addr)))


if __name__ == '__main__':
    binary = sys.argv[1]
    crashlog = sys.argv[2]
    offsets = json.loads(sys.argv[3]) if len(sys.argv) > 3 else None

    with open(crashlog, 'r') as file:
        data = file.read()

    p = CrashLogPatcher(data, binary, offsets)
    p.patch_executable()
    p.patch_uuid()
    p.patch_addresses()

    with open(crashlog, 'w') as file:
        file.write(p.data)
