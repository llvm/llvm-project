// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+lse128 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+lse128 < %s \
// RUN:        | llvm-objdump -d --mattr=+lse128 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+lse128 < %s \
// RUN:        | llvm-objdump -d --mattr=-lse128 - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+lse128 < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+lse128 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST



ldclrp   x1, x2, [x11]
// CHECK-INST: ldclrp x1, x2, [x11]
// CHECK-ENCODING: encoding: [0x61,0x11,0x22,0x19]
// CHECK-ERROR: :[[@LINE-3]]:1: error: instruction requires: lse128
// CHECK-UNKNOWN:  19221161 <unknown>

ldclrp   x21, x22, [sp]
// CHECK-INST: ldclrp x21, x22, [sp]
// CHECK-ENCODING: encoding: [0xf5,0x13,0x36,0x19]
// CHECK-ERROR: :[[@LINE-3]]:1: error: instruction requires: lse128
// CHECK-UNKNOWN:  193613f5 <unknown>

ldclrpa  x1, x2, [x11]
// CHECK-INST: ldclrpa x1, x2, [x11]
// CHECK-ENCODING: encoding: [0x61,0x11,0xa2,0x19]
// CHECK-ERROR: :[[@LINE-3]]:1: error: instruction requires: lse128
// CHECK-UNKNOWN:  19a21161 <unknown>

ldclrpa  x21, x22, [sp]
// CHECK-INST: ldclrpa x21, x22, [sp]
// CHECK-ENCODING: encoding: [0xf5,0x13,0xb6,0x19]
// CHECK-ERROR: :[[@LINE-3]]:1: error: instruction requires: lse128
// CHECK-UNKNOWN:  19b613f5 <unknown>

ldclrpal x1, x2, [x11]
// CHECK-INST: ldclrpal x1, x2, [x11]
// CHECK-ENCODING: encoding: [0x61,0x11,0xe2,0x19]
// CHECK-ERROR: :[[@LINE-3]]:1: error: instruction requires: lse128
// CHECK-UNKNOWN:  19e21161 <unknown>

ldclrpal x21, x22, [sp]
// CHECK-INST: ldclrpal x21, x22, [sp]
// CHECK-ENCODING: encoding: [0xf5,0x13,0xf6,0x19]
// CHECK-ERROR: :[[@LINE-3]]:1: error: instruction requires: lse128
// CHECK-UNKNOWN:  19f613f5 <unknown>

ldclrpl  x1, x2, [x11]
// CHECK-INST: ldclrpl x1, x2, [x11]
// CHECK-ENCODING: encoding: [0x61,0x11,0x62,0x19]
// CHECK-ERROR: :[[@LINE-3]]:1: error: instruction requires: lse128
// CHECK-UNKNOWN:  19621161 <unknown>

ldclrpl  x21, x22, [sp]
// CHECK-INST: ldclrpl x21, x22, [sp]
// CHECK-ENCODING: encoding: [0xf5,0x13,0x76,0x19]
// CHECK-ERROR: :[[@LINE-3]]:1: error: instruction requires: lse128
// CHECK-UNKNOWN:  197613f5 <unknown>

ldsetp   x1, x2, [x11]
// CHECK-INST: ldsetp x1, x2, [x11]
// CHECK-ENCODING: encoding: [0x61,0x31,0x22,0x19]
// CHECK-ERROR: :[[@LINE-3]]:1: error: instruction requires: lse128
// CHECK-UNKNOWN:  19223161 <unknown>

ldsetp   x21, x22, [sp]
// CHECK-INST: ldsetp x21, x22, [sp]
// CHECK-ENCODING: encoding: [0xf5,0x33,0x36,0x19]
// CHECK-ERROR: :[[@LINE-3]]:1: error: instruction requires: lse128
// CHECK-UNKNOWN:  193633f5 <unknown>

ldsetpa  x1, x2, [x11]
// CHECK-INST: ldsetpa x1, x2, [x11]
// CHECK-ENCODING: encoding: [0x61,0x31,0xa2,0x19]
// CHECK-ERROR: :[[@LINE-3]]:1: error: instruction requires: lse128
// CHECK-UNKNOWN:  19a23161 <unknown>

ldsetpa  x21, x22, [sp]
// CHECK-INST: ldsetpa x21, x22, [sp]
// CHECK-ENCODING: encoding: [0xf5,0x33,0xb6,0x19]
// CHECK-ERROR: :[[@LINE-3]]:1: error: instruction requires: lse128
// CHECK-UNKNOWN:  19b633f5 <unknown>

ldsetpal x1, x2, [x11]
// CHECK-INST: ldsetpal x1, x2, [x11]
// CHECK-ENCODING: encoding: [0x61,0x31,0xe2,0x19]
// CHECK-ERROR: :[[@LINE-3]]:1: error: instruction requires: lse128
// CHECK-UNKNOWN:  19e23161 <unknown>

ldsetpal x21, x22, [sp]
// CHECK-INST: ldsetpal x21, x22, [sp]
// CHECK-ENCODING: encoding: [0xf5,0x33,0xf6,0x19]
// CHECK-ERROR: :[[@LINE-3]]:1: error: instruction requires: lse128
// CHECK-UNKNOWN:  19f633f5 <unknown>

ldsetpl  x1, x2, [x11]
// CHECK-INST: ldsetpl x1, x2, [x11]
// CHECK-ENCODING: encoding: [0x61,0x31,0x62,0x19]
// CHECK-ERROR: :[[@LINE-3]]:1: error: instruction requires: lse128
// CHECK-UNKNOWN:  19623161 <unknown>

ldsetpl  x21, x22, [sp]
// CHECK-INST: ldsetpl x21, x22, [sp]
// CHECK-ENCODING: encoding: [0xf5,0x33,0x76,0x19]
// CHECK-ERROR: :[[@LINE-3]]:1: error: instruction requires: lse128
// CHECK-UNKNOWN:  197633f5 <unknown>

swpp     x1, x2, [x11]
// CHECK-INST: swpp x1, x2, [x11]
// CHECK-ENCODING: encoding: [0x61,0x81,0x22,0x19]
// CHECK-ERROR: :[[@LINE-3]]:1: error: instruction requires: lse128
// CHECK-UNKNOWN:  19228161 <unknown>

swpp     x21, x22, [sp]
// CHECK-INST: swpp x21, x22, [sp]
// CHECK-ENCODING: encoding: [0xf5,0x83,0x36,0x19]
// CHECK-ERROR: :[[@LINE-3]]:1: error: instruction requires: lse128
// CHECK-UNKNOWN:  193683f5 <unknown>

swppa    x1, x2, [x11]
// CHECK-INST: swppa x1, x2, [x11]
// CHECK-ENCODING: encoding: [0x61,0x81,0xa2,0x19]
// CHECK-ERROR: :[[@LINE-3]]:1: error: instruction requires: lse128
// CHECK-UNKNOWN:  19a28161 <unknown>

swppa    x21, x22, [sp]
// CHECK-INST: swppa x21, x22, [sp]
// CHECK-ENCODING: encoding: [0xf5,0x83,0xb6,0x19]
// CHECK-ERROR: :[[@LINE-3]]:1: error: instruction requires: lse128
// CHECK-UNKNOWN:  19b683f5 <unknown>

swppal   x1, x2, [x11]
// CHECK-INST: swppal x1, x2, [x11]
// CHECK-ENCODING: encoding: [0x61,0x81,0xe2,0x19]
// CHECK-ERROR: :[[@LINE-3]]:1: error: instruction requires: lse128
// CHECK-UNKNOWN:  19e28161 <unknown>

swppal   x21, x22, [sp]
// CHECK-INST: swppal x21, x22, [sp]
// CHECK-ENCODING: encoding: [0xf5,0x83,0xf6,0x19]
// CHECK-ERROR: :[[@LINE-3]]:1: error: instruction requires: lse128
// CHECK-UNKNOWN:  19f683f5 <unknown>

swppl    x1, x2, [x11]
// CHECK-INST: swppl x1, x2, [x11]
// CHECK-ENCODING: encoding: [0x61,0x81,0x62,0x19]
// CHECK-ERROR: :[[@LINE-3]]:1: error: instruction requires: lse128
// CHECK-UNKNOWN:  19628161 <unknown>

swppl    x21, x22, [sp]
// CHECK-INST: swppl x21, x22, [sp]
// CHECK-ENCODING: encoding: [0xf5,0x83,0x76,0x19]
// CHECK-ERROR: :[[@LINE-3]]:1: error: instruction requires: lse128
// CHECK-UNKNOWN:  197683f5 <unknown>
