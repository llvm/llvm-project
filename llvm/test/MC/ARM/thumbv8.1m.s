// RUN: not llvm-mc -triple=thumbv8.1m.main-none-eabi -show-encoding < %s 2>%t \
// RUN:   | FileCheck --check-prefix=CHECK %s
// RUN:     FileCheck --check-prefix=ERROR < %t %s
// RUN: not llvm-mc -triple=thumbv8.1m.main-none-eabi -mattr=+fp-armv8d16sp,+fullfp16 -show-encoding < %s 2>%t \
// RUN:   | FileCheck --check-prefix=CHECK-FP %s
// RUN:     FileCheck --check-prefix=ERROR-FP < %t %s
// RUN: not llvm-mc -triple=thumbv8.1m.main-none-eabi -mattr=-lob -show-encoding < %s 2>%t \
// RUN:   | FileCheck --check-prefix=CHECK-NOLOB %s
// RUN:     FileCheck --check-prefix=ERROR-NOLOB < %t %s

// Check that .arm is invalid
// ERROR: target does not support ARM mode
// ERROR-FP: target does not support ARM mode
.arm

// Make sure the addition of CLRM does not mess up normal register lists
// ERROR: invalid operand for instruction
// ERROR-FP: invalid operand for instruction
push {r0, apsr}

// Instruction availibility checks

// 'Branch Future and Low Overhead Loop instructions'

// For tests where the LOB extension is turned off, we can't always
// depend on the nice diagnostic 'error: instruction requires: lob',
// because if AsmMatcher can find anything else wrong with the
// instruction, it won't report a specific cause of failure ('multiple
// types of mismatch, so not reporting near-miss'). This can happen in
// the error cases below where the instruction deliberately has
// something else wrong with it, and it can also happen when the
// instruction takes a condition-code argument, because with LOB
// turned off, the operand parsing will reinterpret 'eq' or 'ne' or
// similar as a SymbolRef, and then it won't even match against
// MCK_CondCodeNoAL. So that counts as a second cause of failure from
// AsmMatcher's point of view as well. Hence, a lot of the NOLOB error
// checks just check for "error:", enforcing that MC found *something*
// wrong with the instruction.

// ERROR: :[[@LINE+3]]:{{[0-9]+}}: error: branch location out of range or not a multiple of 2
// ERROR-FP: :[[@LINE+2]]:{{[0-9]+}}: error: branch location out of range or not a multiple of 2
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error:
bf #-2, #10

// ERROR: :[[@LINE+3]]:{{[0-9]+}}: error: branch location out of range or not a multiple of 2
// ERROR-FP: :[[@LINE+2]]:{{[0-9]+}}: error: branch location out of range or not a multiple of 2
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error:
bf #0, #10

// ERROR: :[[@LINE+3]]:{{[0-9]+}}: error: branch location out of range or not a multiple of 2
// ERROR-FP: :[[@LINE+2]]:{{[0-9]+}}: error: branch location out of range or not a multiple of 2
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error:
bf #7, #10

// ERROR: :[[@LINE+3]]:{{[0-9]+}}: error: branch location out of range or not a multiple of 2
// ERROR-FP: :[[@LINE+2]]:{{[0-9]+}}: error: branch location out of range or not a multiple of 2
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error:
bfx #-4, r3

// ERROR: :[[@LINE+3]]:{{[0-9]+}}: error: branch location out of range or not a multiple of 2
// ERROR-FP: :[[@LINE+2]]:{{[0-9]+}}: error: branch location out of range or not a multiple of 2
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error:
bfx #0, r3

// ERROR: :[[@LINE+3]]:{{[0-9]+}}: error: branch location out of range or not a multiple of 2
// ERROR-FP: :[[@LINE+2]]:{{[0-9]+}}: error: branch location out of range or not a multiple of 2
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error:
bfx #13, r3

// ERROR: :[[@LINE+3]]:{{[0-9]+}}: error: branch location out of range or not a multiple of 2
// ERROR-FP: :[[@LINE+2]]:{{[0-9]+}}: error: branch location out of range or not a multiple of 2
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error:
bfl #-2, #20

// ERROR: :[[@LINE+3]]:{{[0-9]+}}: error: branch location out of range or not a multiple of 2
// ERROR-FP: :[[@LINE+2]]:{{[0-9]+}}: error: branch location out of range or not a multiple of 2
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error:
bfl #0, #20

// ERROR: :[[@LINE+3]]:{{[0-9]+}}: error: branch location out of range or not a multiple of 2
// ERROR-FP: :[[@LINE+2]]:{{[0-9]+}}: error: branch location out of range or not a multiple of 2
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error:
bfl #13, #20

// ERROR: :[[@LINE+3]]:{{[0-9]+}}: error: branch target out of range or not a multiple of 2
// ERROR-FP: :[[@LINE+2]]:{{[0-9]+}}: error: branch target out of range or not a multiple of 2
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error:
bf #4, #65536

// ERROR: :[[@LINE+3]]:{{[0-9]+}}: error: branch target out of range or not a multiple of 2
// ERROR-FP: :[[@LINE+2]]:{{[0-9]+}}: error: branch target out of range or not a multiple of 2
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error:
bf #4, #-65538

// CHECK: bf #4, #0
// CHECK-FP: bf #4, #0
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error:
bf #4, #0

// ERROR: :[[@LINE+3]]:{{[0-9]+}}: error: branch target out of range or not a multiple of 2
// ERROR-FP: :[[@LINE+2]]:{{[0-9]+}}: error: branch target out of range or not a multiple of 2
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error:
bfl #4, #262144

// ERROR: :[[@LINE+3]]:{{[0-9]+}}: error: branch target out of range or not a multiple of 2
// ERROR-FP: :[[@LINE+2]]:{{[0-9]+}}: error: branch target out of range or not a multiple of 2
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error:
bfl #4, #-262146

// CHECK: bfl #4, #0
// CHECK-FP: bfl #4, #0
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error:
bfl #4, #0

// ERROR: :[[@LINE+3]]:{{[0-9]+}}: error: branch location out of range or not a multiple of 2
// ERROR-FP: :[[@LINE+2]]:{{[0-9]+}}: error: branch location out of range or not a multiple of 2
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error:
bfcsel #-2, #10, #2, eq

// ERROR: :[[@LINE+3]]:{{[0-9]+}}: error: branch location out of range or not a multiple of 2
// ERROR-FP: :[[@LINE+2]]:{{[0-9]+}}: error: branch location out of range or not a multiple of 2
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error:
bfcsel #0, #10, #2, eq

// ERROR: :[[@LINE+3]]:{{[0-9]+}}: error: branch location out of range or not a multiple of 2
// ERROR-FP: :[[@LINE+2]]:{{[0-9]+}}: error: branch location out of range or not a multiple of 2
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error:
bfcsel #13, #10, #15, eq

// ERROR: :[[@LINE+3]]:{{[0-9]+}}: error: branch target out of range or not a multiple of 2
// ERROR-FP: :[[@LINE+2]]:{{[0-9]+}}: error: branch target out of range or not a multiple of 2
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error:
bfcsel #4, #65536, #6, eq

// ERROR: :[[@LINE+3]]:{{[0-9]+}}: error: branch target out of range or not a multiple of 2
// ERROR-FP: :[[@LINE+2]]:{{[0-9]+}}: error: branch target out of range or not a multiple of 2
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error:
bfcsel #4, #-65538, #8, eq

// ERROR: :[[@LINE+3]]:{{[0-9]+}}: error: else branch target must be 2 or 4 greater than the branch location
// ERROR-FP: :[[@LINE+2]]:{{[0-9]+}}: error: else branch target must be 2 or 4 greater than the branch location
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error:
bfcsel #4, #65534, #10, eq

// CHECK: bfcsel #4, #0, #8, eq
// CHECK-FP: bfcsel #4, #0, #8, eq
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error:
bfcsel #4, #0, #8, eq

// CHECK: bf  .Lbranch, .Ltarget      @ encoding: [0x40'B',0xf0'B',0x01'B',0xe0'B']
// CHECK-FP: bf  .Lbranch, .Ltarget      @ encoding: [0x40'B',0xf0'B',0x01'B',0xe0'B']
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
bf  .Lbranch, .Ltarget

// CHECK: bfcsel  .Lbranch, .Lthen, .Lelse, ne @ encoding: [0x04'C',0xf0'C',0x01'C',0xe0'C']
// CHECK-FP: bfcsel  .Lbranch, .Lthen, .Lelse, ne @ encoding: [0x04'C',0xf0'C',0x01'C',0xe0'C']
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error:
bfcsel  .Lbranch, .Lthen, .Lelse, ne

// CHECK: bfx .Lbranch, r3            @ encoding: [0x63'A',0xf0'A',0x01'A',0xe0'A']
// CHECK-FP: bfx .Lbranch, r3            @ encoding: [0x63'A',0xf0'A',0x01'A',0xe0'A']
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
bfx .Lbranch, r3

// CHECK: bfl .Lbranch, .Ltarget      @ encoding: [B,0xf0'B',0x01'B',0xc0'B']
// CHECK-FP: bfl .Lbranch, .Ltarget      @ encoding: [B,0xf0'B',0x01'B',0xc0'B']
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
bfl .Lbranch, .Ltarget

// CHECK: bflx .Lbranch, r7           @ encoding: [0x77'A',0xf0'A',0x01'A',0xe0'A']
// CHECK-FP: bflx .Lbranch, r7           @ encoding: [0x77'A',0xf0'A',0x01'A',0xe0'A']
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
bflx .Lbranch, r7

// CHECK: wls lr, r2, .Lend           @ encoding: [0x42'A',0xf0'A',0x01'A',0xc0'A']
// CHECK-FP: wls lr, r2, .Lend           @ encoding: [0x42'A',0xf0'A',0x01'A',0xc0'A']
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
wls lr, r2, .Lend

// CHECK: wls lr, r2, #0
// CHECK-FP: wls lr, r2, #0
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
wls lr, r2, #0

// CHECK: dls lr, r2                  @ encoding: [0x42,0xf0,0x01,0xe0]
// CHECK-FP: dls lr, r2                  @ encoding: [0x42,0xf0,0x01,0xe0]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
dls lr, r2

// CHECK: le lr, .Lstart              @ encoding: [0x0f'A',0xf0'A',0x01'A',0xc0'A']
// CHECK-FP: le lr, .Lstart              @ encoding: [0x0f'A',0xf0'A',0x01'A',0xc0'A']
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
le lr, .Lstart

// CHECK: le  .Lstart                 @ encoding: [0x2f'A',0xf0'A',0x01'A',0xc0'A']
// CHECK-FP: le  .Lstart                 @ encoding: [0x2f'A',0xf0'A',0x01'A',0xc0'A']
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
le .Lstart

// CHECK: dls     lr, lr  @ encoding: [0x4e,0xf0,0x01,0xe0]
// CHECK-FP: dls     lr, lr  @ encoding: [0x4e,0xf0,0x01,0xe0]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
dls     lr, lr

// CHECK: dls     lr, r0  @ encoding: [0x40,0xf0,0x01,0xe0]
// CHECK-FP: dls     lr, r0  @ encoding: [0x40,0xf0,0x01,0xe0]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
dls     lr, r0

// CHECK: dls     lr, r1  @ encoding: [0x41,0xf0,0x01,0xe0]
// CHECK-FP: dls     lr, r1  @ encoding: [0x41,0xf0,0x01,0xe0]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
dls     lr, r1

// CHECK: dls     lr, r10  @ encoding: [0x4a,0xf0,0x01,0xe0]
// CHECK-FP: dls     lr, r10  @ encoding: [0x4a,0xf0,0x01,0xe0]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
dls     lr, r10

// CHECK: dls     lr, r11  @ encoding: [0x4b,0xf0,0x01,0xe0]
// CHECK-FP: dls     lr, r11  @ encoding: [0x4b,0xf0,0x01,0xe0]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
dls     lr, r11

// CHECK: dls     lr, r12  @ encoding: [0x4c,0xf0,0x01,0xe0]
// CHECK-FP: dls     lr, r12  @ encoding: [0x4c,0xf0,0x01,0xe0]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
dls     lr, r12

// CHECK: dls     lr, r2  @ encoding: [0x42,0xf0,0x01,0xe0]
// CHECK-FP: dls     lr, r2  @ encoding: [0x42,0xf0,0x01,0xe0]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
dls     lr, r2

// CHECK: dls     lr, r3  @ encoding: [0x43,0xf0,0x01,0xe0]
// CHECK-FP: dls     lr, r3  @ encoding: [0x43,0xf0,0x01,0xe0]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
dls     lr, r3

// CHECK: dls     lr, r5  @ encoding: [0x45,0xf0,0x01,0xe0]
// CHECK-FP: dls     lr, r5  @ encoding: [0x45,0xf0,0x01,0xe0]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
dls     lr, r5

// CHECK: dls     lr, r6  @ encoding: [0x46,0xf0,0x01,0xe0]
// CHECK-FP: dls     lr, r6  @ encoding: [0x46,0xf0,0x01,0xe0]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
dls     lr, r6

// CHECK: dls     lr, r7  @ encoding: [0x47,0xf0,0x01,0xe0]
// CHECK-FP: dls     lr, r7  @ encoding: [0x47,0xf0,0x01,0xe0]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
dls     lr, r7

// CHECK: dls     lr, r8  @ encoding: [0x48,0xf0,0x01,0xe0]
// CHECK-FP: dls     lr, r8  @ encoding: [0x48,0xf0,0x01,0xe0]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
dls     lr, r8

// CHECK: dls     lr, r9  @ encoding: [0x49,0xf0,0x01,0xe0]
// CHECK-FP: dls     lr, r9  @ encoding: [0x49,0xf0,0x01,0xe0]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
dls     lr, r9

// CHECK: le      #-106  @ encoding: [0x2f,0xf0,0x35,0xc8]
// CHECK-FP: le      #-106  @ encoding: [0x2f,0xf0,0x35,0xc8]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
le      #-106

// CHECK: le      #-1172  @ encoding: [0x2f,0xf0,0x4b,0xc2]
// CHECK-FP: le      #-1172  @ encoding: [0x2f,0xf0,0x4b,0xc2]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
le      #-1172

// CHECK: le      #-1210  @ encoding: [0x2f,0xf0,0x5d,0xca]
// CHECK-FP: le      #-1210  @ encoding: [0x2f,0xf0,0x5d,0xca]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
le      #-1210

// CHECK: le      #-1260  @ encoding: [0x2f,0xf0,0x77,0xc2]
// CHECK-FP: le      #-1260  @ encoding: [0x2f,0xf0,0x77,0xc2]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
le      #-1260

// CHECK: le      #-1262  @ encoding: [0x2f,0xf0,0x77,0xca]
// CHECK-FP: le      #-1262  @ encoding: [0x2f,0xf0,0x77,0xca]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
le      #-1262

// CHECK: le      #-1284  @ encoding: [0x2f,0xf0,0x83,0xc2]
// CHECK-FP: le      #-1284  @ encoding: [0x2f,0xf0,0x83,0xc2]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
le      #-1284

// CHECK: le      #-1286  @ encoding: [0x2f,0xf0,0x83,0xca]
// CHECK-FP: le      #-1286  @ encoding: [0x2f,0xf0,0x83,0xca]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
le      #-1286

// CHECK: le      #-1556  @ encoding: [0x2f,0xf0,0x0b,0xc3]
// CHECK-FP: le      #-1556  @ encoding: [0x2f,0xf0,0x0b,0xc3]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
le      #-1556

// CHECK: le      #-178  @ encoding: [0x2f,0xf0,0x59,0xc8]
// CHECK-FP: le      #-178  @ encoding: [0x2f,0xf0,0x59,0xc8]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
le      #-178

// CHECK: le      #-1882  @ encoding: [0x2f,0xf0,0xad,0xcb]
// CHECK-FP: le      #-1882  @ encoding: [0x2f,0xf0,0xad,0xcb]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
le      #-1882

// CHECK: le      #-1900  @ encoding: [0x2f,0xf0,0xb7,0xc3]
// CHECK-FP: le      #-1900  @ encoding: [0x2f,0xf0,0xb7,0xc3]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
le      #-1900

// CHECK: le      #-1910  @ encoding: [0x2f,0xf0,0xbb,0xcb]
// CHECK-FP: le      #-1910  @ encoding: [0x2f,0xf0,0xbb,0xcb]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
le      #-1910

// CHECK: le      #-2076  @ encoding: [0x2f,0xf0,0x0f,0xc4]
// CHECK-FP: le      #-2076  @ encoding: [0x2f,0xf0,0x0f,0xc4]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
le      #-2076

// CHECK: le      #-2266  @ encoding: [0x2f,0xf0,0x6d,0xcc]
// CHECK-FP: le      #-2266  @ encoding: [0x2f,0xf0,0x6d,0xcc]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
le      #-2266

// CHECK: le      #-2324  @ encoding: [0x2f,0xf0,0x8b,0xc4]
// CHECK-FP: le      #-2324  @ encoding: [0x2f,0xf0,0x8b,0xc4]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
le      #-2324

// CHECK: le      #-2328  @ encoding: [0x2f,0xf0,0x8d,0xc4]
// CHECK-FP: le      #-2328  @ encoding: [0x2f,0xf0,0x8d,0xc4]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
le      #-2328

// CHECK: le      #-2456  @ encoding: [0x2f,0xf0,0xcd,0xc4]
// CHECK-FP: le      #-2456  @ encoding: [0x2f,0xf0,0xcd,0xc4]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
le      #-2456

// CHECK: le      #-246  @ encoding: [0x2f,0xf0,0x7b,0xc8]
// CHECK-FP: le      #-246  @ encoding: [0x2f,0xf0,0x7b,0xc8]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
le      #-246

// CHECK: le      #-2476  @ encoding: [0x2f,0xf0,0xd7,0xc4]
// CHECK-FP: le      #-2476  @ encoding: [0x2f,0xf0,0xd7,0xc4]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
le      #-2476

// CHECK: le      #-2578  @ encoding: [0x2f,0xf0,0x09,0xcd]
// CHECK-FP: le      #-2578  @ encoding: [0x2f,0xf0,0x09,0xcd]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
le      #-2578

// CHECK: le      #-262  @ encoding: [0x2f,0xf0,0x83,0xc8]
// CHECK-FP: le      #-262  @ encoding: [0x2f,0xf0,0x83,0xc8]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
le      #-262

// CHECK: le      #-2660  @ encoding: [0x2f,0xf0,0x33,0xc5]
// CHECK-FP: le      #-2660  @ encoding: [0x2f,0xf0,0x33,0xc5]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
le      #-2660

// CHECK: le      #-2722  @ encoding: [0x2f,0xf0,0x51,0xcd]
// CHECK-FP: le      #-2722  @ encoding: [0x2f,0xf0,0x51,0xcd]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
le      #-2722

// CHECK: le      #-2868  @ encoding: [0x2f,0xf0,0x9b,0xc5]
// CHECK-FP: le      #-2868  @ encoding: [0x2f,0xf0,0x9b,0xc5]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
le      #-2868

// CHECK: le      #-2882  @ encoding: [0x2f,0xf0,0xa1,0xcd]
// CHECK-FP: le      #-2882  @ encoding: [0x2f,0xf0,0xa1,0xcd]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
le      #-2882

// CHECK: le      #-3154  @ encoding: [0x2f,0xf0,0x29,0xce]
// CHECK-FP: le      #-3154  @ encoding: [0x2f,0xf0,0x29,0xce]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
le      #-3154

// CHECK: le      #-3274  @ encoding: [0x2f,0xf0,0x65,0xce]
// CHECK-FP: le      #-3274  @ encoding: [0x2f,0xf0,0x65,0xce]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
le      #-3274

// CHECK: le      #-3352  @ encoding: [0x2f,0xf0,0x8d,0xc6]
// CHECK-FP: le      #-3352  @ encoding: [0x2f,0xf0,0x8d,0xc6]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
le      #-3352

// CHECK: le      #-338  @ encoding: [0x2f,0xf0,0xa9,0xc8]
// CHECK-FP: le      #-338  @ encoding: [0x2f,0xf0,0xa9,0xc8]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
le      #-338

// CHECK: le      #-3458  @ encoding: [0x2f,0xf0,0xc1,0xce]
// CHECK-FP: le      #-3458  @ encoding: [0x2f,0xf0,0xc1,0xce]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
le      #-3458

// CHECK: le      #-3480  @ encoding: [0x2f,0xf0,0xcd,0xc6]
// CHECK-FP: le      #-3480  @ encoding: [0x2f,0xf0,0xcd,0xc6]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
le      #-3480

// CHECK: le      #-3542  @ encoding: [0x2f,0xf0,0xeb,0xce]
// CHECK-FP: le      #-3542  @ encoding: [0x2f,0xf0,0xeb,0xce]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
le      #-3542

// CHECK: le      #-3644  @ encoding: [0x2f,0xf0,0x1f,0xc7]
// CHECK-FP: le      #-3644  @ encoding: [0x2f,0xf0,0x1f,0xc7]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
le      #-3644

// CHECK: le      #-3676  @ encoding: [0x2f,0xf0,0x2f,0xc7]
// CHECK-FP: le      #-3676  @ encoding: [0x2f,0xf0,0x2f,0xc7]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
le      #-3676

// CHECK: le      #-3692  @ encoding: [0x2f,0xf0,0x37,0xc7]
// CHECK-FP: le      #-3692  @ encoding: [0x2f,0xf0,0x37,0xc7]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
le      #-3692

// CHECK: le      #-3860  @ encoding: [0x2f,0xf0,0x8b,0xc7]
// CHECK-FP: le      #-3860  @ encoding: [0x2f,0xf0,0x8b,0xc7]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
le      #-3860

// CHECK: le      #-3986  @ encoding: [0x2f,0xf0,0xc9,0xcf]
// CHECK-FP: le      #-3986  @ encoding: [0x2f,0xf0,0xc9,0xcf]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
le      #-3986

// CHECK: le      #-4006  @ encoding: [0x2f,0xf0,0xd3,0xcf]
// CHECK-FP: le      #-4006  @ encoding: [0x2f,0xf0,0xd3,0xcf]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
le      #-4006

// CHECK: le      #-4034  @ encoding: [0x2f,0xf0,0xe1,0xcf]
// CHECK-FP: le      #-4034  @ encoding: [0x2f,0xf0,0xe1,0xcf]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
le      #-4034

// CHECK: le      #-4060  @ encoding: [0x2f,0xf0,0xef,0xc7]
// CHECK-FP: le      #-4060  @ encoding: [0x2f,0xf0,0xef,0xc7]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
le      #-4060

// CHECK: le      #-4068  @ encoding: [0x2f,0xf0,0xf3,0xc7]
// CHECK-FP: le      #-4068  @ encoding: [0x2f,0xf0,0xf3,0xc7]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
le      #-4068

// CHECK: le      #-478  @ encoding: [0x2f,0xf0,0xef,0xc8]
// CHECK-FP: le      #-478  @ encoding: [0x2f,0xf0,0xef,0xc8]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
le      #-478

// CHECK: le      #-544  @ encoding: [0x2f,0xf0,0x11,0xc1]
// CHECK-FP: le      #-544  @ encoding: [0x2f,0xf0,0x11,0xc1]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
le      #-544

// CHECK: le      #-586  @ encoding: [0x2f,0xf0,0x25,0xc9]
// CHECK-FP: le      #-586  @ encoding: [0x2f,0xf0,0x25,0xc9]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
le      #-586

// CHECK: le      #-606  @ encoding: [0x2f,0xf0,0x2f,0xc9]
// CHECK-FP: le      #-606  @ encoding: [0x2f,0xf0,0x2f,0xc9]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
le      #-606

// CHECK: le      #-656  @ encoding: [0x2f,0xf0,0x49,0xc1]
// CHECK-FP: le      #-656  @ encoding: [0x2f,0xf0,0x49,0xc1]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
le      #-656

// CHECK: le      #-740  @ encoding: [0x2f,0xf0,0x73,0xc1]
// CHECK-FP: le      #-740  @ encoding: [0x2f,0xf0,0x73,0xc1]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
le      #-740

// CHECK: le      #-762  @ encoding: [0x2f,0xf0,0x7d,0xc9]
// CHECK-FP: le      #-762  @ encoding: [0x2f,0xf0,0x7d,0xc9]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
le      #-762

// CHECK: le      #-862  @ encoding: [0x2f,0xf0,0xaf,0xc9]
// CHECK-FP: le      #-862  @ encoding: [0x2f,0xf0,0xaf,0xc9]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
le      #-862

// CHECK: le      #-870  @ encoding: [0x2f,0xf0,0xb3,0xc9]
// CHECK-FP: le      #-870  @ encoding: [0x2f,0xf0,0xb3,0xc9]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
le      #-870

// CHECK: le      lr, #-1080  @ encoding: [0x0f,0xf0,0x1d,0xc2]
// CHECK-FP: le      lr, #-1080  @ encoding: [0x0f,0xf0,0x1d,0xc2]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
le      lr, #-1080

// CHECK: le      lr, #-1104  @ encoding: [0x0f,0xf0,0x29,0xc2]
// CHECK-FP: le      lr, #-1104  @ encoding: [0x0f,0xf0,0x29,0xc2]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
le      lr, #-1104

// CHECK: le      lr, #-1152  @ encoding: [0x0f,0xf0,0x41,0xc2]
// CHECK-FP: le      lr, #-1152  @ encoding: [0x0f,0xf0,0x41,0xc2]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
le      lr, #-1152

// CHECK: le      lr, #-1462  @ encoding: [0x0f,0xf0,0xdb,0xca]
// CHECK-FP: le      lr, #-1462  @ encoding: [0x0f,0xf0,0xdb,0xca]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
le      lr, #-1462

// CHECK: le      lr, #-1470  @ encoding: [0x0f,0xf0,0xdf,0xca]
// CHECK-FP: le      lr, #-1470  @ encoding: [0x0f,0xf0,0xdf,0xca]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
le      lr, #-1470

// CHECK: le      lr, #-1612  @ encoding: [0x0f,0xf0,0x27,0xc3]
// CHECK-FP: le      lr, #-1612  @ encoding: [0x0f,0xf0,0x27,0xc3]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
le      lr, #-1612

// CHECK: le      lr, #-1632  @ encoding: [0x0f,0xf0,0x31,0xc3]
// CHECK-FP: le      lr, #-1632  @ encoding: [0x0f,0xf0,0x31,0xc3]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
le      lr, #-1632

// CHECK: le      lr, #-1694  @ encoding: [0x0f,0xf0,0x4f,0xcb]
// CHECK-FP: le      lr, #-1694  @ encoding: [0x0f,0xf0,0x4f,0xcb]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
le      lr, #-1694

// CHECK: le      lr, #-1714  @ encoding: [0x0f,0xf0,0x59,0xcb]
// CHECK-FP: le      lr, #-1714  @ encoding: [0x0f,0xf0,0x59,0xcb]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
le      lr, #-1714

// CHECK: le      lr, #-1850  @ encoding: [0x0f,0xf0,0x9d,0xcb]
// CHECK-FP: le      lr, #-1850  @ encoding: [0x0f,0xf0,0x9d,0xcb]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
le      lr, #-1850

// CHECK: le      lr, #-1878  @ encoding: [0x0f,0xf0,0xab,0xcb]
// CHECK-FP: le      lr, #-1878  @ encoding: [0x0f,0xf0,0xab,0xcb]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
le      lr, #-1878

// CHECK: le      lr, #-1896  @ encoding: [0x0f,0xf0,0xb5,0xc3]
// CHECK-FP: le      lr, #-1896  @ encoding: [0x0f,0xf0,0xb5,0xc3]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
le      lr, #-1896

// CHECK: le      lr, #-1922  @ encoding: [0x0f,0xf0,0xc1,0xcb]
// CHECK-FP: le      lr, #-1922  @ encoding: [0x0f,0xf0,0xc1,0xcb]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
le      lr, #-1922

// CHECK: le      lr, #-1926  @ encoding: [0x0f,0xf0,0xc3,0xcb]
// CHECK-FP: le      lr, #-1926  @ encoding: [0x0f,0xf0,0xc3,0xcb]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
le      lr, #-1926

// CHECK: le      lr, #-2  @ encoding: [0x0f,0xf0,0x01,0xc8]
// CHECK-FP: le      lr, #-2  @ encoding: [0x0f,0xf0,0x01,0xc8]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
le      lr, #-2

// CHECK: le      lr, #-2104  @ encoding: [0x0f,0xf0,0x1d,0xc4]
// CHECK-FP: le      lr, #-2104  @ encoding: [0x0f,0xf0,0x1d,0xc4]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
le      lr, #-2104

// CHECK: le      lr, #-2116  @ encoding: [0x0f,0xf0,0x23,0xc4]
// CHECK-FP: le      lr, #-2116  @ encoding: [0x0f,0xf0,0x23,0xc4]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
le      lr, #-2116

// CHECK: le      lr, #-2144  @ encoding: [0x0f,0xf0,0x31,0xc4]
// CHECK-FP: le      lr, #-2144  @ encoding: [0x0f,0xf0,0x31,0xc4]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
le      lr, #-2144

// CHECK: le      lr, #-2188  @ encoding: [0x0f,0xf0,0x47,0xc4]
// CHECK-FP: le      lr, #-2188  @ encoding: [0x0f,0xf0,0x47,0xc4]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
le      lr, #-2188

// CHECK: le      lr, #-2344  @ encoding: [0x0f,0xf0,0x95,0xc4]
// CHECK-FP: le      lr, #-2344  @ encoding: [0x0f,0xf0,0x95,0xc4]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
le      lr, #-2344

// CHECK: le      lr, #-2456  @ encoding: [0x0f,0xf0,0xcd,0xc4]
// CHECK-FP: le      lr, #-2456  @ encoding: [0x0f,0xf0,0xcd,0xc4]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
le      lr, #-2456

// CHECK: le      lr, #-2608  @ encoding: [0x0f,0xf0,0x19,0xc5]
// CHECK-FP: le      lr, #-2608  @ encoding: [0x0f,0xf0,0x19,0xc5]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
le      lr, #-2608

// CHECK: le      lr, #-2616  @ encoding: [0x0f,0xf0,0x1d,0xc5]
// CHECK-FP: le      lr, #-2616  @ encoding: [0x0f,0xf0,0x1d,0xc5]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
le      lr, #-2616

// CHECK: le      lr, #-2622  @ encoding: [0x0f,0xf0,0x1f,0xcd]
// CHECK-FP: le      lr, #-2622  @ encoding: [0x0f,0xf0,0x1f,0xcd]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
le      lr, #-2622

// CHECK: le      lr, #-2680  @ encoding: [0x0f,0xf0,0x3d,0xc5]
// CHECK-FP: le      lr, #-2680  @ encoding: [0x0f,0xf0,0x3d,0xc5]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
le      lr, #-2680

// CHECK: le      lr, #-2694  @ encoding: [0x0f,0xf0,0x43,0xcd]
// CHECK-FP: le      lr, #-2694  @ encoding: [0x0f,0xf0,0x43,0xcd]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
le      lr, #-2694

// CHECK: le      lr, #-2850  @ encoding: [0x0f,0xf0,0x91,0xcd]
// CHECK-FP: le      lr, #-2850  @ encoding: [0x0f,0xf0,0x91,0xcd]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
le      lr, #-2850

// CHECK: le      lr, #-2860  @ encoding: [0x0f,0xf0,0x97,0xc5]
// CHECK-FP: le      lr, #-2860  @ encoding: [0x0f,0xf0,0x97,0xc5]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
le      lr, #-2860

// CHECK: le      lr, #-3004  @ encoding: [0x0f,0xf0,0xdf,0xc5]
// CHECK-FP: le      lr, #-3004  @ encoding: [0x0f,0xf0,0xdf,0xc5]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
le      lr, #-3004

// CHECK: le      lr, #-3018  @ encoding: [0x0f,0xf0,0xe5,0xcd]
// CHECK-FP: le      lr, #-3018  @ encoding: [0x0f,0xf0,0xe5,0xcd]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
le      lr, #-3018

// CHECK: le      lr, #-304  @ encoding: [0x0f,0xf0,0x99,0xc0]
// CHECK-FP: le      lr, #-304  @ encoding: [0x0f,0xf0,0x99,0xc0]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
le      lr, #-304

// CHECK: le      lr, #-3098  @ encoding: [0x0f,0xf0,0x0d,0xce]
// CHECK-FP: le      lr, #-3098  @ encoding: [0x0f,0xf0,0x0d,0xce]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
le      lr, #-3098

// CHECK: le      lr, #-3228  @ encoding: [0x0f,0xf0,0x4f,0xc6]
// CHECK-FP: le      lr, #-3228  @ encoding: [0x0f,0xf0,0x4f,0xc6]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
le      lr, #-3228

// CHECK: le      lr, #-3316  @ encoding: [0x0f,0xf0,0x7b,0xc6]
// CHECK-FP: le      lr, #-3316  @ encoding: [0x0f,0xf0,0x7b,0xc6]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
le      lr, #-3316

// CHECK: le      lr, #-3332  @ encoding: [0x0f,0xf0,0x83,0xc6]
// CHECK-FP: le      lr, #-3332  @ encoding: [0x0f,0xf0,0x83,0xc6]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
le      lr, #-3332

// CHECK: le      lr, #-3354  @ encoding: [0x0f,0xf0,0x8d,0xce]
// CHECK-FP: le      lr, #-3354  @ encoding: [0x0f,0xf0,0x8d,0xce]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
le      lr, #-3354

// CHECK: le      lr, #-3962  @ encoding: [0x0f,0xf0,0xbd,0xcf]
// CHECK-FP: le      lr, #-3962  @ encoding: [0x0f,0xf0,0xbd,0xcf]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
le      lr, #-3962

// CHECK: le      lr, #-4042  @ encoding: [0x0f,0xf0,0xe5,0xcf]
// CHECK-FP: le      lr, #-4042  @ encoding: [0x0f,0xf0,0xe5,0xcf]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
le      lr, #-4042

// CHECK: le      lr, #-4052  @ encoding: [0x0f,0xf0,0xeb,0xc7]
// CHECK-FP: le      lr, #-4052  @ encoding: [0x0f,0xf0,0xeb,0xc7]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
le      lr, #-4052

// CHECK: le      lr, #-458  @ encoding: [0x0f,0xf0,0xe5,0xc8]
// CHECK-FP: le      lr, #-458  @ encoding: [0x0f,0xf0,0xe5,0xc8]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
le      lr, #-458

// CHECK: le      lr, #-56  @ encoding: [0x0f,0xf0,0x1d,0xc0]
// CHECK-FP: le      lr, #-56  @ encoding: [0x0f,0xf0,0x1d,0xc0]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
le      lr, #-56

// CHECK: le      lr, #-582  @ encoding: [0x0f,0xf0,0x23,0xc9]
// CHECK-FP: le      lr, #-582  @ encoding: [0x0f,0xf0,0x23,0xc9]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
le      lr, #-582

// CHECK: le      lr, #-676  @ encoding: [0x0f,0xf0,0x53,0xc1]
// CHECK-FP: le      lr, #-676  @ encoding: [0x0f,0xf0,0x53,0xc1]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
le      lr, #-676

// CHECK: le      lr, #-752  @ encoding: [0x0f,0xf0,0x79,0xc1]
// CHECK-FP: le      lr, #-752  @ encoding: [0x0f,0xf0,0x79,0xc1]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
le      lr, #-752

// CHECK: le      lr, #-76  @ encoding: [0x0f,0xf0,0x27,0xc0]
// CHECK-FP: le      lr, #-76  @ encoding: [0x0f,0xf0,0x27,0xc0]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
le      lr, #-76

// CHECK: le      lr, #-802  @ encoding: [0x0f,0xf0,0x91,0xc9]
// CHECK-FP: le      lr, #-802  @ encoding: [0x0f,0xf0,0x91,0xc9]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
le      lr, #-802

// CHECK: le      lr, #-862  @ encoding: [0x0f,0xf0,0xaf,0xc9]
// CHECK-FP: le      lr, #-862  @ encoding: [0x0f,0xf0,0xaf,0xc9]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
le      lr, #-862

// CHECK: le      lr, #-902  @ encoding: [0x0f,0xf0,0xc3,0xc9]
// CHECK-FP: le      lr, #-902  @ encoding: [0x0f,0xf0,0xc3,0xc9]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
le      lr, #-902

// CHECK: le      lr, #-968  @ encoding: [0x0f,0xf0,0xe5,0xc1]
// CHECK-FP: le      lr, #-968  @ encoding: [0x0f,0xf0,0xe5,0xc1]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
le      lr, #-968

// CHECK: wls     lr, lr, #1192  @ encoding: [0x4e,0xf0,0x55,0xc2]
// CHECK-FP: wls     lr, lr, #1192  @ encoding: [0x4e,0xf0,0x55,0xc2]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
wls     lr, lr, #1192

// CHECK: wls     lr, lr, #2134  @ encoding: [0x4e,0xf0,0x2b,0xcc]
// CHECK-FP: wls     lr, lr, #2134  @ encoding: [0x4e,0xf0,0x2b,0xcc]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
wls     lr, lr, #2134

// CHECK: wls     lr, lr, #962  @ encoding: [0x4e,0xf0,0xe1,0xc9]
// CHECK-FP: wls     lr, lr, #962  @ encoding: [0x4e,0xf0,0xe1,0xc9]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
wls     lr, lr, #962

// CHECK: wls     lr, r0, #1668  @ encoding: [0x40,0xf0,0x43,0xc3]
// CHECK-FP: wls     lr, r0, #1668  @ encoding: [0x40,0xf0,0x43,0xc3]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
wls     lr, r0, #1668

// CHECK: wls     lr, r0, #2706  @ encoding: [0x40,0xf0,0x49,0xcd]
// CHECK-FP: wls     lr, r0, #2706  @ encoding: [0x40,0xf0,0x49,0xcd]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
wls     lr, r0, #2706

// CHECK: wls     lr, r0, #3026  @ encoding: [0x40,0xf0,0xe9,0xcd]
// CHECK-FP: wls     lr, r0, #3026  @ encoding: [0x40,0xf0,0xe9,0xcd]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
wls     lr, r0, #3026

// CHECK: wls     lr, r0, #3436  @ encoding: [0x40,0xf0,0xb7,0xc6]
// CHECK-FP: wls     lr, r0, #3436  @ encoding: [0x40,0xf0,0xb7,0xc6]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
wls     lr, r0, #3436

// CHECK: wls     lr, r1, #1060  @ encoding: [0x41,0xf0,0x13,0xc2]
// CHECK-FP: wls     lr, r1, #1060  @ encoding: [0x41,0xf0,0x13,0xc2]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
wls     lr, r1, #1060

// CHECK: wls     lr, r1, #4036  @ encoding: [0x41,0xf0,0xe3,0xc7]
// CHECK-FP: wls     lr, r1, #4036  @ encoding: [0x41,0xf0,0xe3,0xc7]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
wls     lr, r1, #4036

// CHECK: wls     lr, r1, #538  @ encoding: [0x41,0xf0,0x0d,0xc9]
// CHECK-FP: wls     lr, r1, #538  @ encoding: [0x41,0xf0,0x0d,0xc9]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
wls     lr, r1, #538

// CHECK: wls     lr, r10, #1404  @ encoding: [0x4a,0xf0,0xbf,0xc2]
// CHECK-FP: wls     lr, r10, #1404  @ encoding: [0x4a,0xf0,0xbf,0xc2]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
wls     lr, r10, #1404

// CHECK: wls     lr, r10, #1408  @ encoding: [0x4a,0xf0,0xc1,0xc2]
// CHECK-FP: wls     lr, r10, #1408  @ encoding: [0x4a,0xf0,0xc1,0xc2]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
wls     lr, r10, #1408

// CHECK: wls     lr, r10, #2358  @ encoding: [0x4a,0xf0,0x9b,0xcc]
// CHECK-FP: wls     lr, r10, #2358  @ encoding: [0x4a,0xf0,0x9b,0xcc]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
wls     lr, r10, #2358

// CHECK: wls     lr, r10, #4086  @ encoding: [0x4a,0xf0,0xfb,0xcf]
// CHECK-FP: wls     lr, r10, #4086  @ encoding: [0x4a,0xf0,0xfb,0xcf]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
wls     lr, r10, #4086

// CHECK: wls     lr, r11, #1442  @ encoding: [0x4b,0xf0,0xd1,0xca]
// CHECK-FP: wls     lr, r11, #1442  @ encoding: [0x4b,0xf0,0xd1,0xca]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
wls     lr, r11, #1442

// CHECK: wls     lr, r11, #2678  @ encoding: [0x4b,0xf0,0x3b,0xcd]
// CHECK-FP: wls     lr, r11, #2678  @ encoding: [0x4b,0xf0,0x3b,0xcd]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
wls     lr, r11, #2678

// CHECK: wls     lr, r11, #3610  @ encoding: [0x4b,0xf0,0x0d,0xcf]
// CHECK-FP: wls     lr, r11, #3610  @ encoding: [0x4b,0xf0,0x0d,0xcf]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
wls     lr, r11, #3610

// CHECK: wls     lr, r12, #206  @ encoding: [0x4c,0xf0,0x67,0xc8]
// CHECK-FP: wls     lr, r12, #206  @ encoding: [0x4c,0xf0,0x67,0xc8]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
wls     lr, r12, #206

// CHECK: wls     lr, r12, #2896  @ encoding: [0x4c,0xf0,0xa9,0xc5]
// CHECK-FP: wls     lr, r12, #2896  @ encoding: [0x4c,0xf0,0xa9,0xc5]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
wls     lr, r12, #2896

// CHECK: wls     lr, r12, #3258  @ encoding: [0x4c,0xf0,0x5d,0xce]
// CHECK-FP: wls     lr, r12, #3258  @ encoding: [0x4c,0xf0,0x5d,0xce]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
wls     lr, r12, #3258

// CHECK: wls     lr, r2, #3242  @ encoding: [0x42,0xf0,0x55,0xce]
// CHECK-FP: wls     lr, r2, #3242  @ encoding: [0x42,0xf0,0x55,0xce]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
wls     lr, r2, #3242

// CHECK: wls     lr, r2, #3832  @ encoding: [0x42,0xf0,0x7d,0xc7]
// CHECK-FP: wls     lr, r2, #3832  @ encoding: [0x42,0xf0,0x7d,0xc7]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
wls     lr, r2, #3832

// CHECK: wls     lr, r2, #872  @ encoding: [0x42,0xf0,0xb5,0xc1]
// CHECK-FP: wls     lr, r2, #872  @ encoding: [0x42,0xf0,0xb5,0xc1]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
wls     lr, r2, #872

// CHECK: wls     lr, r3, #3514  @ encoding: [0x43,0xf0,0xdd,0xce]
// CHECK-FP: wls     lr, r3, #3514  @ encoding: [0x43,0xf0,0xdd,0xce]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
wls     lr, r3, #3514

// CHECK: wls     lr, r3, #3636  @ encoding: [0x43,0xf0,0x1b,0xc7]
// CHECK-FP: wls     lr, r3, #3636  @ encoding: [0x43,0xf0,0x1b,0xc7]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
wls     lr, r3, #3636

// CHECK: wls     lr, r3, #3942  @ encoding: [0x43,0xf0,0xb3,0xcf]
// CHECK-FP: wls     lr, r3, #3942  @ encoding: [0x43,0xf0,0xb3,0xcf]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
wls     lr, r3, #3942

// CHECK: wls     lr, r3, #712  @ encoding: [0x43,0xf0,0x65,0xc1]
// CHECK-FP: wls     lr, r3, #712  @ encoding: [0x43,0xf0,0x65,0xc1]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
wls     lr, r3, #712

// CHECK: wls     lr, r4, #2146  @ encoding: [0x44,0xf0,0x31,0xcc]
// CHECK-FP: wls     lr, r4, #2146  @ encoding: [0x44,0xf0,0x31,0xcc]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
wls     lr, r4, #2146

// CHECK: wls     lr, r4, #2486  @ encoding: [0x44,0xf0,0xdb,0xcc]
// CHECK-FP: wls     lr, r4, #2486  @ encoding: [0x44,0xf0,0xdb,0xcc]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
wls     lr, r4, #2486

// CHECK: wls     lr, r5, #1906  @ encoding: [0x45,0xf0,0xb9,0xcb]
// CHECK-FP: wls     lr, r5, #1906  @ encoding: [0x45,0xf0,0xb9,0xcb]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
wls     lr, r5, #1906

// CHECK: wls     lr, r5, #3396  @ encoding: [0x45,0xf0,0xa3,0xc6]
// CHECK-FP: wls     lr, r5, #3396  @ encoding: [0x45,0xf0,0xa3,0xc6]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
wls     lr, r5, #3396

// CHECK: wls     lr, r6, #3326  @ encoding: [0x46,0xf0,0x7f,0xce]
// CHECK-FP: wls     lr, r6, #3326  @ encoding: [0x46,0xf0,0x7f,0xce]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
wls     lr, r6, #3326

// CHECK: wls     lr, r6, #416  @ encoding: [0x46,0xf0,0xd1,0xc0]
// CHECK-FP: wls     lr, r6, #416  @ encoding: [0x46,0xf0,0xd1,0xc0]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
wls     lr, r6, #416

// CHECK: wls     lr, r6, #422  @ encoding: [0x46,0xf0,0xd3,0xc8]
// CHECK-FP: wls     lr, r6, #422  @ encoding: [0x46,0xf0,0xd3,0xc8]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
wls     lr, r6, #422

// CHECK: wls     lr, r7, #3474  @ encoding: [0x47,0xf0,0xc9,0xce]
// CHECK-FP: wls     lr, r7, #3474  @ encoding: [0x47,0xf0,0xc9,0xce]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
wls     lr, r7, #3474

// CHECK: wls     lr, r7, #3640  @ encoding: [0x47,0xf0,0x1d,0xc7]
// CHECK-FP: wls     lr, r7, #3640  @ encoding: [0x47,0xf0,0x1d,0xc7]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
wls     lr, r7, #3640

// CHECK: wls     lr, r8, #2700  @ encoding: [0x48,0xf0,0x47,0xc5]
// CHECK-FP: wls     lr, r8, #2700  @ encoding: [0x48,0xf0,0x47,0xc5]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
wls     lr, r8, #2700

// CHECK: wls     lr, r9, #1114  @ encoding: [0x49,0xf0,0x2d,0xca]
// CHECK-FP: wls     lr, r9, #1114  @ encoding: [0x49,0xf0,0x2d,0xca]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
wls     lr, r9, #1114

// CHECK: wls     lr, r9, #1984  @ encoding: [0x49,0xf0,0xe1,0xc3]
// CHECK-FP: wls     lr, r9, #1984  @ encoding: [0x49,0xf0,0xe1,0xc3]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
wls     lr, r9, #1984

// CHECK: wls     lr, r9, #3758  @ encoding: [0x49,0xf0,0x57,0xcf]
// CHECK-FP: wls     lr, r9, #3758  @ encoding: [0x49,0xf0,0x57,0xcf]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
wls     lr, r9, #3758

// CHECK: wls     lr, r9, #3796  @ encoding: [0x49,0xf0,0x6b,0xc7]
// CHECK-FP: wls     lr, r9, #3796  @ encoding: [0x49,0xf0,0x6b,0xc7]
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires: lob
wls     lr, r9, #3796

// ERROR: :[[@LINE+3]]:{{[0-9]+}}: error: invalid operand for instruction
// ERROR-FP: :[[@LINE+2]]:{{[0-9]+}}: error: invalid operand for instruction
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error:
wls r10, r9, #2

// ERROR: :[[@LINE+3]]:{{[0-9]+}}: error: loop end is out of range or not a positive multiple of 2
// ERROR-FP: :[[@LINE+2]]:{{[0-9]+}}: error: loop end is out of range or not a positive multiple of 2
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error:
wls lr, r9, #1

// ERROR: :[[@LINE+3]]:{{[0-9]+}}: error: loop end is out of range or not a positive multiple of 2
// ERROR-FP: :[[@LINE+2]]:{{[0-9]+}}: error: loop end is out of range or not a positive multiple of 2
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error:
wls lr, r9, #-2

// ERROR: :[[@LINE+3]]:{{[0-9]+}}: error: loop end is out of range or not a positive multiple of 2
// ERROR-FP: :[[@LINE+2]]:{{[0-9]+}}: error: loop end is out of range or not a positive multiple of 2
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error:
wls lr, r9, #4096

// ERROR: :[[@LINE+3]]:{{[0-9]+}}: error: loop start is out of range or not a negative multiple of 2
// ERROR-FP: :[[@LINE+2]]:{{[0-9]+}}: error: loop start is out of range or not a negative multiple of 2
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error:
le lr, #-1

// ERROR: :[[@LINE+3]]:{{[0-9]+}}: error: loop start is out of range or not a negative multiple of 2
// ERROR-FP: :[[@LINE+2]]:{{[0-9]+}}: error: loop start is out of range or not a negative multiple of 2
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error:
le lr, #2

// ERROR: :[[@LINE+3]]:{{[0-9]+}}: error: loop start is out of range or not a negative multiple of 2
// ERROR-FP: :[[@LINE+2]]:{{[0-9]+}}: error: loop start is out of range or not a negative multiple of 2
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}: error:
le lr, #-4096

// ERROR: :[[@LINE+3]]:{{[0-9]+}}: error: invalid operand for instruction
// ERROR-FP: :[[@LINE+2]]:{{[0-9]+}}: error: invalid operand for instruction
// ERROR-NOLOB: :[[@LINE+1]]:{{[0-9]+}}:
le r10, #-4

# ERROR: :[[@LINE+3]]:{{[0-9]+}}: error: instruction requires: 16-bit fp registers
# ERROR-NOLOB: :[[@LINE+2]]:{{[0-9]+}}: error: instruction requires: 16-bit fp registers
# CHECK-FP: vmov.f16 s7, r8 @ encoding: [0x03,0xee,0x90,0x89]
vmov.f16 s7, r8

# ERROR: :[[@LINE+3]]:{{[0-9]+}}: error: instruction requires: 16-bit fp registers
# ERROR-NOLOB: :[[@LINE+2]]:{{[0-9]+}}: error: instruction requires: 16-bit fp registers
# CHECK-FP: vmov.f16 s10, r5 @ encoding: [0x05,0xee,0x10,0x59]
vmov.f16 s10, r5

# ERROR: :[[@LINE+3]]:{{[0-9]+}}: error: instruction requires: 16-bit fp registers
# ERROR-NOLOB: :[[@LINE+2]]:{{[0-9]+}}: error: instruction requires: 16-bit fp registers
# CHECK-FP: vmov.f16 s31, r10 @ encoding: [0x0f,0xee,0x90,0xa9]
vmov.f16 s31, r10

# ERROR: :[[@LINE+3]]:{{[0-9]+}}: error: instruction requires: 16-bit fp registers
# ERROR-NOLOB: :[[@LINE+2]]:{{[0-9]+}}: error: instruction requires: 16-bit fp registers
# CHECK-FP: vmov.f16 r8, s7 @ encoding: [0x13,0xee,0x90,0x89]
vmov.f16 r8, s7

# ERROR: :[[@LINE+3]]:{{[0-9]+}}: error: instruction requires: 16-bit fp registers
# ERROR-NOLOB: :[[@LINE+2]]:{{[0-9]+}}: error: instruction requires: 16-bit fp registers
# CHECK-FP: vmov.f16 r5, s10 @ encoding: [0x15,0xee,0x10,0x59]
vmov.f16 r5, s10

# ERROR: :[[@LINE+3]]:{{[0-9]+}}: error: instruction requires: 16-bit fp registers
# ERROR-NOLOB: :[[@LINE+2]]:{{[0-9]+}}: error: instruction requires: 16-bit fp registers
# CHECK-FP: vmov.f16 r10, s31        @ encoding: [0x1f,0xee,0x90,0xa9]
vmov.f16 r10, s31

# ERROR: :[[@LINE+3]]:{{[0-9]+}}: error: invalid instruction
# ERROR-NOLOB: :[[@LINE+2]]:{{[0-9]+}}: error: invalid instruction
# ERROR-FP: operand must be a register in range [r0, r12] or r14
vmov.f16 sp, s10

# ERROR: :[[@LINE+3]]:{{[0-9]+}}: error: invalid instruction
# ERROR-NOLOB: :[[@LINE+2]]:{{[0-9]+}}: error: invalid instruction
# ERROR-FP: operand must be a register in range [r0, r12] or r14
vmov.f16 s10, sp

# ERROR: :[[@LINE+3]]:{{[0-9]+}}: error: invalid instruction
# ERROR-NOLOB: :[[@LINE+2]]:{{[0-9]+}}: error: invalid instruction
# ERROR-FP: operand must be a register in range [s0, s31]
vmov.f16 r10, d1

# ERROR: :[[@LINE+3]]:{{[0-9]+}}: error: invalid instruction
# ERROR-NOLOB: :[[@LINE+2]]:{{[0-9]+}}: error: invalid instruction
# ERROR-FP: operand must be a register in range [s0, s31]
vmov.f16 r10, s32

# ERROR: :[[@LINE+3]]:{{[0-9]+}}: error: invalid instruction
# ERROR-NOLOB: :[[@LINE+2]]:{{[0-9]+}}: error: invalid instruction
# ERROR-FP: operand must be a register in range [s0, s31]
vmov.f16 d1, r10

# ERROR: :[[@LINE+3]]:{{[0-9]+}}: error: invalid instruction
# ERROR-NOLOB: :[[@LINE+2]]:{{[0-9]+}}: error: invalid instruction
# ERROR-FP: operand must be a register in range [s0, s31]
vmov.f16 s32, r10

# CHECK: cinc    lr, r2, lo  @ encoding: [0x52,0xea,0x22,0x9e]
# CHECK-FP: cinc    lr, r2, lo  @ encoding: [0x52,0xea,0x22,0x9e]
# CHECK-NOLOB: cinc    lr, r2, lo  @ encoding: [0x52,0xea,0x22,0x9e]
csinc   lr, r2, r2, hs

# CHECK: cinc    lr, r7, pl  @ encoding: [0x57,0xea,0x47,0x9e]
# CHECK-FP: cinc    lr, r7, pl  @ encoding: [0x57,0xea,0x47,0x9e]
# CHECK-NOLOB: cinc    lr, r7, pl  @ encoding: [0x57,0xea,0x47,0x9e]
cinc    lr, r7, pl

# CHECK: cinv    lr, r12, hs  @ encoding: [0x5c,0xea,0x3c,0xae]
# CHECK-FP: cinv    lr, r12, hs  @ encoding: [0x5c,0xea,0x3c,0xae]
# CHECK-NOLOB: cinv    lr, r12, hs  @ encoding: [0x5c,0xea,0x3c,0xae]
cinv    lr, r12, hs

# CHECK: cneg    lr, r10, hs  @ encoding: [0x5a,0xea,0x3a,0xbe]
# CHECK-FP: cneg    lr, r10, hs  @ encoding: [0x5a,0xea,0x3a,0xbe]
# CHECK-NOLOB: cneg    lr, r10, hs  @ encoding: [0x5a,0xea,0x3a,0xbe]
csneg   lr, r10, r10, lo

# CHECK: csel    r9, r9, r11, vc  @ encoding: [0x59,0xea,0x7b,0x89]
# CHECK-FP: csel    r9, r9, r11, vc  @ encoding: [0x59,0xea,0x7b,0x89]
# CHECK-NOLOB: csel    r9, r9, r11, vc  @ encoding: [0x59,0xea,0x7b,0x89]
csel    r9, r9, r11, vc

# CHECK: cset    lr, eq  @ encoding: [0x5f,0xea,0x1f,0x9e]
# CHECK-FP: cset    lr, eq  @ encoding: [0x5f,0xea,0x1f,0x9e]
# CHECK-NOLOB: cset    lr, eq  @ encoding: [0x5f,0xea,0x1f,0x9e]
cset    lr, eq

# CHECK: csetm   lr, hs  @ encoding: [0x5f,0xea,0x3f,0xae]
# CHECK-FP: csetm   lr, hs  @ encoding: [0x5f,0xea,0x3f,0xae]
# CHECK-NOLOB: csetm   lr, hs  @ encoding: [0x5f,0xea,0x3f,0xae]
csetm   lr, hs

# CHECK: csinc   lr, r10, r7, le  @ encoding: [0x5a,0xea,0xd7,0x9e]
# CHECK-FP: csinc   lr, r10, r7, le  @ encoding: [0x5a,0xea,0xd7,0x9e]
# CHECK-NOLOB: csinc   lr, r10, r7, le  @ encoding: [0x5a,0xea,0xd7,0x9e]
csinc   lr, r10, r7, le

# CHECK: csinv   lr, r5, zr, hs  @ encoding: [0x55,0xea,0x2f,0xae]
# CHECK-FP: csinv   lr, r5, zr, hs  @ encoding: [0x55,0xea,0x2f,0xae]
# CHECK-NOLOB: csinv   lr, r5, zr, hs  @ encoding: [0x55,0xea,0x2f,0xae]
csinv   lr, r5, zr, hs

# CHECK: cinv    lr, r2, pl  @ encoding: [0x52,0xea,0x42,0xae]
# CHECK-FP: cinv    lr, r2, pl  @ encoding: [0x52,0xea,0x42,0xae]
# CHECK-NOLOB: cinv    lr, r2, pl  @ encoding: [0x52,0xea,0x42,0xae]
csinv   lr, r2, r2, mi

# CHECK: csel r0, r0, r1, eq @ encoding: [0x50,0xea,0x01,0x80]
# CHECK-FP: csel r0, r0, r1, eq @ encoding: [0x50,0xea,0x01,0x80]
# CHECK-NOLOB: csel r0, r0, r1, eq @ encoding: [0x50,0xea,0x01,0x80]
csel r0, r0, r1, eq

// ERROR: :[[@LINE+1]]:{{[0-9]+}}: error: operand must be a register in range [r0, r12] or r14
csel sp, r0, r1, eq
// ERROR: :[[@LINE+1]]:{{[0-9]+}}: error: operand must be a register in range [r0, r12] or r14
csel pc, r0, r1, eq

// ERROR: :[[@LINE+1]]:{{[0-9]+}}: error: operand must be a register in range [r0, r12] or r14 or zr
csel r0, sp, r1, eq
// ERROR: :[[@LINE+1]]:{{[0-9]+}}: error: operand must be a register in range [r0, r12] or r14 or zr
csel r0, pc, r1, eq

// ERROR: :[[@LINE+1]]:{{[0-9]+}}: error: operand must be a register in range [r0, r12] or r14 or zr
csinc r0, sp, r1, eq
// ERROR: :[[@LINE+1]]:{{[0-9]+}}: error: operand must be a register in range [r0, r12] or r14 or zr
csinc r0, pc, r1, eq

// ERROR: :[[@LINE+1]]:{{[0-9]+}}: error: operand must be a register in range [r0, r12] or r14 or zr
csinv r0, sp, r1, eq
// ERROR: :[[@LINE+1]]:{{[0-9]+}}: error: operand must be a register in range [r0, r12] or r14 or zr
csinv r0, pc, r1, eq

// ERROR: :[[@LINE+1]]:{{[0-9]+}}: error: operand must be a register in range [r0, r12] or r14 or zr
csneg r0, sp, r1, eq
// ERROR: :[[@LINE+1]]:{{[0-9]+}}: error: operand must be a register in range [r0, r12] or r14 or zr
csneg r0, pc, r1, eq

// ERROR: :[[@LINE+1]]:{{[0-9]+}}: error: operand must be a register in range [r0, r12] or r14 or zr
csel r0, r0, sp, eq
// ERROR: :[[@LINE+1]]:{{[0-9]+}}: error: operand must be a register in range [r0, r12] or r14 or zr
csel r0, r0, pc, eq

// ERROR: :[[@LINE+2]]:{{[0-9]+}}: error: instructions in IT block must be predicable
it eq
csel r0, r0, r1, eq

// ERROR: :[[@LINE+2]]:{{[0-9]+}}: error: instructions in IT block must be predicable
it eq
csinc r0, r0, r1, ne

// ERROR: :[[@LINE+2]]:{{[0-9]+}}: error: instructions in IT block must be predicable
it gt
csinv r0, r0, r1, ge

// ERROR: :[[@LINE+2]]:{{[0-9]+}}: error: instructions in IT block must be predicable
it lt
csneg r0, r0, r1, gt
