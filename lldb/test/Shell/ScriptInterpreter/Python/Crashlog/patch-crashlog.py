#!/usr/bin/env python

import json
import os
import re
import subprocess
import sys
import argparse


class CrashLogPatcher:

    SYMBOL_REGEX = re.compile(r'^([0-9a-fA-F]+) T _(.*)$')
    UUID_REGEX = re.compile(r'UUID: ([-0-9a-fA-F]+) \(([^\(]+)\) .*')

    def __init__(self, data, binary, offsets, json):
        self.data = data
        self.binary = binary
        self.offsets = offsets
        self.json = json

    def patch_executable(self):
        self.data = self.data.replace("@EXEC@", self.binary)
        self.data = self.data.replace("@NAME@", os.path.basename(self.binary))

    def patch_uuid(self):
        output = subprocess.check_output(['dwarfdump', '--uuid', self.binary]).decode("utf-8")
        m = self.UUID_REGEX.match(output)
        if m:
            self.data = self.data.replace("@UUID@", m.group(1))

    def patch_addresses(self):
        if not self.offsets:
            return
        output = subprocess.check_output(['nm', self.binary]).decode("utf-8")
        for line in output.splitlines():
            m = self.SYMBOL_REGEX.match(line)
            if m:
                address = m.group(1)
                symbol = m.group(2)
                if symbol in self.offsets:
                    patch_addr = int(m.group(1), 16) + int(
                        self.offsets[symbol])
                    if self.json:
                        patch_addr = patch_addr - 0x100000000
                        representation = int
                    else:
                        representation = hex
                    self.data = self.data.replace(
                        "@{}@".format(symbol), str(representation(patch_addr)))

    def remove_metadata(self):
        self.data= self.data[self.data.index('\n') + 1:]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Crashlog Patcher')
    parser.add_argument('--binary', required=True)
    parser.add_argument('--crashlog', required=True)
    parser.add_argument('--offsets', required=True)
    parser.add_argument('--json', default=False, action='store_true')
    parser.add_argument('--no-metadata', default=False, action='store_true')
    args = parser.parse_args()

    offsets = json.loads(args.offsets)

    with open(args.crashlog, 'r') as file:
        data = file.read()

    p = CrashLogPatcher(data, args.binary, offsets, args.json)
    p.patch_executable()
    p.patch_uuid()
    p.patch_addresses()

    if args.no_metadata:
        p.remove_metadata()

    with open(args.crashlog, 'w') as file:
        file.write(p.data)
