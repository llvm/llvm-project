# RUN: llvm-mc %s -show-encoding -triple=mips-unknown-linux-gnu \
# RUN:   -mcpu=mips32r6 | FileCheck %s
# RUN: llvm-mc %s -show-encoding -triple=mips64-unknown-linux-gnu \
# RUN:   -mcpu=mips64r6 | FileCheck %s

  .set crc
  crc32b	$1, $2, $1 	# CHECK: crc32b	$1, $2, $1 # encoding: [0x7c,0x41,0x00,0x0f]
