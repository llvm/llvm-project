#!/bin/bash
#=============================== test_xteams.sh -=============================//
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===----------------------------------------------------------------------===//
# 
#  test_xteams.sh: Script to test high performance Cross Team Scan functions
#                  in llvm-project/openmp/libomptarget/DeviceRTL/src/Xteams.cpp
#                  It compiles and executes test_xteams.cpp in 4 configs of
#                  device threads 1024, 512, 256 and 128.
#
#  See README file in this directory for more information.
#
#===----------------------------------------------------------------------===//

LLVM_INSTALL=${LLVM_INSTALL:-$HOME/rocm/aomp}
[ ! -f $LLVM_INSTALL/bin/clang ] && echo "ERROR: no LLVM install at $LLVM_INSTALL" && exit 1

OFFLOAD_ARCH=${OFFLOAD_ARCH:-gfx906}

tmpdir=/tmp/$USER/xteams && mkdir -p $tmpdir
[ ! -d $tmpdir ] && echo "ERROR: could not create $tmpdir"

# ARRAY_SIZE=${ARRAY_SIZE:-1000000}
ARRAY_SIZE=${ARRAY_SIZE:-41943040}
as_arg="-D_ARRAY_SIZE=$ARRAY_SIZE"

NUM_TEAMS=${NUM_TEAMS:-80}

nt_args="-D_XTEAM_NUM_THREADS=1024 -D_XTEAM_NUM_TEAMS=$NUM_TEAMS"
echo " COMPILE with --offload-arch=$OFFLOAD_ARCH $as_arg $nt_args"
$LLVM_INSTALL/bin/clang++ -O0 -I. $as_arg $nt_args -fopenmp --offload-arch=$OFFLOAD_ARCH test_xteams.cpp -o $tmpdir/xteams_1024 -lstdc++ -latomic
rc01=$?

nt_args="-D_XTEAM_NUM_THREADS=512 -D_XTEAM_NUM_TEAMS=$NUM_TEAMS"
echo " COMPILE with --offload-arch=$OFFLOAD_ARCH $as_arg $nt_args"
$LLVM_INSTALL/bin/clang++ -O0 -I. $as_arg $nt_args -fopenmp --offload-arch=$OFFLOAD_ARCH test_xteams.cpp -o $tmpdir/xteams_512 -lstdc++ -latomic
rc11=$?

nt_args="-D_XTEAM_NUM_THREADS=256 -D_XTEAM_NUM_TEAMS=$NUM_TEAMS"
echo " COMPILE with --offload-arch=$OFFLOAD_ARCH $as_arg $nt_args"
$LLVM_INSTALL/bin/clang++ -O0 -I. $as_arg $nt_args -fopenmp --offload-arch=$OFFLOAD_ARCH test_xteams.cpp -o $tmpdir/xteams_256 -lstdc++ -latomic
rc21=$?

nt_args="-D_XTEAM_NUM_THREADS=128 -D_XTEAM_NUM_TEAMS=$NUM_TEAMS"
echo " COMPILE with --offload-arch=$OFFLOAD_ARCH $as_arg $nt_args"
$LLVM_INSTALL/bin/clang++ -O0 -I. $as_arg $nt_args -fopenmp --offload-arch=$OFFLOAD_ARCH test_xteams.cpp -o $tmpdir/xteams_128 -lstdc++ -latomic
rc31=$?

[ $rc01 == 0 ] && echo "START EXECUTE xteams_1024" && $tmpdir/xteams_1024 > $tmpdir/xteams_1024.out
rc02=$?
[ $rc11 == 0 ] && echo "START EXECUTE xteams_512" && $tmpdir/xteams_512 > $tmpdir/xteams_512.out
rc12=$?
[ $rc21 == 0 ] && echo "START EXECUTE xteams_256" && $tmpdir/xteams_256 > $tmpdir/xteams_256.out
rc22=$?
[ $rc31 == 0 ] && echo "START EXECUTE xteams_128" && $tmpdir/xteams_128 > $tmpdir/xteams_128.out
rc32=$?


echo 
rc=$(( $rc01 + $rc02 + $rc11 + $rc12 + $rc21 + $rc22 + $rc31 + $rc32 ))
if [ $rc != 0 ] ; then 
  echo "ERRORS DETECTED!"
else
  echo "No errors detected"
fi
echo "Logs and binaries saved to $tmpdir"
exit $rc
