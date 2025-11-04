#! /usr/bin/python3

import os
import sys
import argparse
import pathlib

llvmsrcroot = os.path.normpath(f"{__file__}/../../..") # Adapt to location in source tree
sys.path.insert(0, os.path.join(llvmsrcroot, '.buildbot/common'))
import worker


# For information
for k,v in os.environ.items():
    print(f"{k}={v}")

def relative_if_possible(path, relative_to):
    path = os.path.normpath(path)
    try:
        result = os.path.relpath(path, start=relative_to)
        return result if result else path
    except Exception:
        return  path


parser = argparse.ArgumentParser()
parser.add_argument('--cachefile', default=relative_if_possible(pathlib.Path(__file__).with_suffix('.cmake'), llvmsrcroot), help='CMake cache seed')
parser.add_argument('--jobs', '-j', help='Override the number fo default jobs')
parser.add_argument('--clean', type=bool, default=os.environ.get('BUILDBOT_CLEAN'), help='Whether to delete source-, install-, and build-dirs before running')
parser.add_argument('--clobber', type=bool, default=os.environ.get('BUILDBOT_CLOBBER'), help='Whether to delete install- and build-dirs before running')
args, _ = parser.parse_known_args()


cwd = os.getcwd()





buildbot_buildername = os.environ.get('BUILDBOT_BUILDERNAME')
buildbot_revision = os.environ.get('BUILDBOT_REVISION', 'origin/main')


os.environ['NINJA_STATUS'] = "[%p/%es :: %u->%r->%f (of %t)] "

llvmbuilddir = "build-llvm"
testsuitesrcdir = "testsuite.src"
testsuitebuilddir = "build-testsuite"
llvminstalldir = 'install-llvm'
print(f"Using build directory: {cwd}")

# NEVER clean llvmsrcroot or cwd!
worker.clean_on_request(args, always=[llvminstalldir,testsuitebuilddir],on_clobber=[llvmbuilddir],on_clean=[testsuitesrcdir])


with worker.step('configure-llvm', halt_on_fail=True):
    cmd = ['cmake',
        '-S', os.path.join(llvmsrcroot,'llvm'),
        '-B', llvmbuilddir,
        '-G', 'Ninja',
        '-C', os.path.join(llvmsrcroot, args.cachefile),
        f'-DCMAKE_INSTALL_PREFIX={llvminstalldir}'
    ]
    if args.jobs:
        cmd.append(f'-DLLVM_LIT_ARGS=-svj{args.jobs}')
    worker.run_command(cmd)

with worker.step('build-llvm', halt_on_fail=True):
    worker.run_command(['ninja', '-C', llvmbuilddir])

with worker.step('check-polly'):
    worker.run_command(['ninja', '-C', llvmbuilddir, 'check-polly'])

with worker. step('install-llvm', halt_on_fail=True):
    worker.run_command(['ninja', '-C', llvmbuilddir, 'install'])

with worker. step('clone-testsuite', halt_on_fail=True):
    worker.checkout('https://github.com/llvm/llvm-test-suite',testsuitesrcdir)

with worker.step('configure-testsuite', halt_on_fail=True):
    cmd = ['cmake',
        '-S', testsuitesrcdir,
        '-B', testsuitebuilddir,
        '-G', 'Ninja',
        '-C', os.path.join(llvmsrcroot, args.cachefile),
        '-DCMAKE_BUILD_TYPE=Release',
        f'-DCMAKE_C_COMPILER={os.path.abspath(llvminstalldir)}/bin/clang',
        f'-DCMAKE_CXX_COMPILER={os.path.abspath(llvminstalldir)}/bin/clang++',
        f'-DCMAKE_C_FLAGS=-mllvm -polly',
        f'-DCMAKE_CXX_FLAGS=-mllvm -polly',
    ]
    if args.jobs:
        cmd.append(f'-DLLVM_LIT_ARGS=-svj{args.jobs}')
    worker.run_command(cmd)

with worker.step('build-testsuite', halt_on_fail=True):
    worker. run_ninja(args, ['-C', testsuitebuilddir])

with worker.step('check-testsuite'):
    worker.run_ninja(args, ['-C', testsuitebuilddir, 'check'])

