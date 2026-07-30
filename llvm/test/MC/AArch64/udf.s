# RUN: llvm-mc -assemble -show-encoding -triple=aarch64- %s | FileCheck %s
# CHECK:  .text
# CHECK-NEXT: udf #0      // encoding: [0x00,0x00,0x00,0x00]
# CHECK-NEXT: udf #513    // encoding: [0x01,0x02,0x00,0x00]
# CHECK-NEXT: udf #65535  // encoding: [0xff,0xff,0x00,0x00]
.text
udf 0
udf 513
udf 65535
