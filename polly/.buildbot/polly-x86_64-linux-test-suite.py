#! /usr/bin/python3

import os
import sys
import pathlib

# Adapt to location in source tree
llvmsrcroot = os.path.normpath(f"{__file__}/../../..")

sys.path.insert(0, os.path.join(llvmsrcroot, '.buildbot/common'))
import worker



parser = worker.common_init()
parser.add_argument('--cachefile', default=worker.relative_if_possible(pathlib.Path(__file__).with_suffix('.cmake'), llvmsrcroot), help='CMake cache seed')
args, _ = parser.parse_known_args()

if args.workdir:
 os.chdir(args.workdir)

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
    worker.run_ninja(args, ['-C', llvmbuilddir], ccache_stats=True)

with worker.step('check-polly'):
    worker.run_ninja(args, ['-C', llvmbuilddir, 'check-polly'], ccache_stats=True)

with worker.step('install-llvm', halt_on_fail=True):
    worker.run_ninja(args, ['-C', llvmbuilddir, 'install'], ccache_stats=True)

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
        f'-DTEST_SUITE_LIT={os.path.abspath(llvmbuilddir)}/bin/llvm-lit',
        f'-DTEST_SUITE_LLVM_SIZE={os.path.abspath(llvmbuilddir)}/bin/llvm-size',
        "-DTEST_SUITE_EXTRA_C_FLAGS=-Wno-unused-command-line-argument -mllvm -polly",
        "-DTEST_SUITE_EXTRA_CXX_FLAGS=-Wno-unused-command-line-argument -mllvm -polly",
    ]
    if args.jobs:
        cmd.append(f'-DLLVM_LIT_ARGS=-svj{args.jobs};-o;report.json')
    worker.run_command(cmd)

with worker.step('build-testsuite', halt_on_fail=True):
    worker. run_ninja(args, ['-C', testsuitebuilddir], ccache_stats=True)

with worker.step('check-testsuite'):
    worker.run_ninja(args, ['-C', testsuitebuilddir, 'check'], ccache_stats=True)

