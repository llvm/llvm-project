# RUN: not llvm-mc -filetype=obj -triple=x86_64-linux-gnu %s 2>&1 | FileCheck %s

       .balign 128 
label00:
// CHECK: value of 253 is too large for field of 1 byte.
  jecxz   label01
// CHECK: value of 251 is too large for field of 1 byte.
  jrcxz   label01
// CHECK: value of 249 is too large for field of 1 byte.
  loop  label01
// CHECK: value of 247 is too large for field of 1 byte. 
  loope  label01
// CHECK: value of 245 is too large for field of 1 byte.
  loopne  label01
        .balign 256 
label01:
// CHECK: value of -259 is too large for field of 1 byte.
  jecxz   label00
// CHECK: value of -261 is too large for field of 1 byte.
  jrcxz   label00
// CHECK: value of -263 is too large for field of 1 byte.
  loop  label00
// CHECK: value of -265 is too large for field of 1 byte.
  loope  label00
// CHECK: value of -267 is too large for field of 1 byte.
  loopne  label00
