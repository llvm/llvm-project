# RUN: llvm-mc %s -triple=mips64-unknown-freebsd -show-encoding 2>%t0 \
# RUN:     | FileCheck %s
# RUN: FileCheck -check-prefix=WARNING %s < %t0
#
# RUN: llvm-mc %s -triple=mips64-unknown-freebsd -show-encoding \
# RUN:     -target-abi n32 2>%t1 | FileCheck %s
# RUN: FileCheck -check-prefix=WARNING %s < %t1
#
# Check that the register names are mapped to their correct numbers for n32/n64
# Second byte of addiu with $zero at rt contains the number of the source
# register.

.set noat
daddiu	$zero, $zero, 0     # CHECK: encoding: [0x64,0x00,0x00,0x00]
daddiu	$at, $zero, 0       # CHECK: encoding: [0x64,0x01,0x00,0x00]
daddiu	$AT, $zero, 0       # CHECK: encoding: [0x64,0x01,0x00,0x00]
daddiu	$v0, $zero, 0       # CHECK: encoding: [0x64,0x02,0x00,0x00]
daddiu	$v1, $zero, 0       # CHECK: encoding: [0x64,0x03,0x00,0x00]
daddiu	$a0, $zero, 0       # CHECK: encoding: [0x64,0x04,0x00,0x00]
daddiu	$a1, $zero, 0       # CHECK: encoding: [0x64,0x05,0x00,0x00]
daddiu	$a2, $zero, 0       # CHECK: encoding: [0x64,0x06,0x00,0x00]
daddiu	$a3, $zero, 0       # CHECK: encoding: [0x64,0x07,0x00,0x00]
daddiu	$a4, $zero, 0       # CHECK: encoding: [0x64,0x08,0x00,0x00]
daddiu	$a5, $zero, 0       # CHECK: encoding: [0x64,0x09,0x00,0x00]
daddiu	$a6, $zero, 0       # CHECK: encoding: [0x64,0x0a,0x00,0x00]
daddiu	$a7, $zero, 0       # CHECK: encoding: [0x64,0x0b,0x00,0x00]
daddiu	$t0, $zero, 0 # [*] # CHECK: encoding: [0x64,0x0c,0x00,0x00]
daddiu	$t1, $zero, 0 # [*] # CHECK: encoding: [0x64,0x0d,0x00,0x00]
daddiu	$t2, $zero, 0 # [*] # CHECK: encoding: [0x64,0x0e,0x00,0x00]
daddiu	$t3, $zero, 0 # [*] # CHECK: encoding: [0x64,0x0f,0x00,0x00]
# WARNING: mips64-register-names-n32-n64.s:[[@LINE+4]]:9: warning: register names $t4-$t7 are only available in O32.
# WARNING-NEXT: daddiu  $t4, $zero, 0       # {{CHECK}}: encoding: [0x64,0x0c,0x00,0x00]
# WARNING-NEXT:          ^~
# WARNING-NEXT:          Did you mean $t0?
daddiu	$t4, $zero, 0       # CHECK: encoding: [0x64,0x0c,0x00,0x00]
# WARNING: mips64-register-names-n32-n64.s:[[@LINE+4]]:9: warning: register names $t4-$t7 are only available in O32.
# WARNING-NEXT: daddiu  $t5, $zero, 0       # {{CHECK}}: encoding: [0x64,0x0d,0x00,0x00]
# WARNING-NEXT:          ^~
# WARNING-NEXT:          Did you mean $t1?
daddiu	$t5, $zero, 0       # CHECK: encoding: [0x64,0x0d,0x00,0x00]
# WARNING: mips64-register-names-n32-n64.s:[[@LINE+4]]:9: warning: register names $t4-$t7 are only available in O32.
# WARNING-NEXT: daddiu  $t6, $zero, 0       # {{CHECK}}: encoding: [0x64,0x0e,0x00,0x00]
# WARNING-NEXT:          ^~
# WARNING-NEXT:          Did you mean $t2?
daddiu	$t6, $zero, 0       # CHECK: encoding: [0x64,0x0e,0x00,0x00]
# WARNING: mips64-register-names-n32-n64.s:[[@LINE+4]]:9: warning: register names $t4-$t7 are only available in O32.
# WARNING-NEXT: daddiu  $t7, $zero, 0       # {{CHECK}}: encoding: [0x64,0x0f,0x00,0x00]
# WARNING-NEXT:          ^~
# WARNING-NEXT:          Did you mean $t3?
daddiu	$t7, $zero, 0       # CHECK: encoding: [0x64,0x0f,0x00,0x00]
daddiu	$s0, $zero, 0       # CHECK: encoding: [0x64,0x10,0x00,0x00]
daddiu	$s1, $zero, 0       # CHECK: encoding: [0x64,0x11,0x00,0x00]
daddiu	$s2, $zero, 0       # CHECK: encoding: [0x64,0x12,0x00,0x00]
daddiu	$s3, $zero, 0       # CHECK: encoding: [0x64,0x13,0x00,0x00]
daddiu	$s4, $zero, 0       # CHECK: encoding: [0x64,0x14,0x00,0x00]
daddiu	$s5, $zero, 0       # CHECK: encoding: [0x64,0x15,0x00,0x00]
daddiu	$s6, $zero, 0       # CHECK: encoding: [0x64,0x16,0x00,0x00]
daddiu	$s7, $zero, 0       # CHECK: encoding: [0x64,0x17,0x00,0x00]
daddiu	$t8, $zero, 0       # CHECK: encoding: [0x64,0x18,0x00,0x00]
daddiu	$t9, $zero, 0       # CHECK: encoding: [0x64,0x19,0x00,0x00]
daddiu	$kt0, $zero, 0      # CHECK: encoding: [0x64,0x1a,0x00,0x00]
daddiu	$kt1, $zero, 0      # CHECK: encoding: [0x64,0x1b,0x00,0x00]
daddiu	$gp, $zero, 0       # CHECK: encoding: [0x64,0x1c,0x00,0x00]
daddiu	$sp, $zero, 0       # CHECK: encoding: [0x64,0x1d,0x00,0x00]
daddiu	$s8, $zero, 0       # CHECK: encoding: [0x64,0x1e,0x00,0x00]
daddiu	$fp, $zero, 0       # CHECK: encoding: [0x64,0x1e,0x00,0x00]
daddiu	$ra, $zero, 0       # CHECK: encoding: [0x64,0x1f,0x00,0x00]

# [*] - t0-t3 are aliases of t4-t7 for compatibility with both the original
#       ABI documentation (using t4-t7) and GNU As (using t0-t3)
