# RUN: not llvm-mc %s -triple=mips-unknown-linux-gnu -show-encoding \
# RUN:     -mcpu=mips32r6 -mattr=+crc 2>%t1
# RUN: FileCheck %s < %t1
# RUN: not llvm-mc %s -triple=mips64-unknown-linux-gnu -show-encoding \
# RUN:     -mcpu=mips64r6 -mattr=+crc 2>%t1
# RUN: FileCheck %s < %t1

  .set nocrc
  crc32b	$1, $2, $1 	# CHECK: instruction requires a CPU feature not currently enabled
