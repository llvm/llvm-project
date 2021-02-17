#!/usr/bin/env python3

"""Helps manage sysroots."""

import argparse
import os
import subprocess
import sys


def make_fake_sysroot(out_dir):
    def cmdout(cmd):
        return subprocess.check_output(cmd).decode(sys.stdout.encoding).strip()

    if sys.platform == 'win32':
        p = os.getenv('ProgramFiles(x86)', 'C:\\Program Files (x86)')

        winsdk = os.getenv('WindowsSdkDir')
        if not winsdk:
            winsdk = os.path.join(p, 'Windows Kits', '10')
            print('%WindowsSdkDir% not set. You might want to run this from')
            print('a Visual Studio cmd prompt. Defaulting to', winsdk)

        vswhere = os.path.join(
                p, 'Microsoft Visual Studio', 'Installer', 'vswhere')
        vcid = 'Microsoft.VisualStudio.Component.VC.Tools.x86.x64'
        vsinstalldir = cmdout(
                [vswhere, '-latest', '-products', '*', '-requires', vcid,
                    '-property', 'installationPath'])

        def mkjunction(dst, src):
            subprocess.check_call(['mklink', '/j', dst, src], shell=True)
        os.mkdir(out_dir)
        mkjunction(os.path.join(out_dir, 'VC'),
                   os.path.join(vsinstalldir, 'VC'))
        os.mkdir(os.path.join(out_dir, 'Windows Kits'))
        mkjunction(os.path.join(out_dir, 'Windows Kits', '10'), winsdk)
    else:
        assert False, "FIXME: Implement on non-win"

    print('Done.')
    if sys.platform == 'win32':
        # CMake doesn't like backslashes in commandline args.
        abs_out_dir = os.path.abspath(out_dir).replace(os.path.sep, '/')
        print('Pass -DLLVM_WINSYSROOT=' + abs_out_dir + ' to cmake.')
    else:
        print('Pass -DCMAKE_SYSROOT=' + abs_out_dir + ' to cmake.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    subparsers = parser.add_subparsers(dest='command', required=True)

    makefake = subparsers.add_parser('make-fake',
            help='Create a sysroot that symlinks to local directories.')
    makefake.add_argument('--out-dir', required=True)

    args = parser.parse_args()

    assert args.command == 'make-fake'
    make_fake_sysroot(args.out_dir)


if __name__ == '__main__':
    main()
