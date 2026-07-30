// RUN: llvm-mc -triple i386-unknown-unknown %s | FileCheck %s

// .exitm is encountered in a normal macro expansion
.macro REP
.rept 3
.long 0
.exitm
.endr
.endm
REP
// Only the output from the first rept expansion should make it through:
// CHECK: .long 0
// CHECK-NOT: .long 0

// .exitm is in a true branch
.macro A
.if 1
.long 1
.exitm
.endif
.long 1
.endm
A
// CHECK: .long 1
// CHECK-NOT: .long 1

// .exitm is in a false branch
.macro B
.if 1
.long 2
.else
.exitm
.endif
.long 2
.endm
B
// CHECK: .long 2
// CHECK: .long 2


// .exitm is in a false branch that is encountered prior to the true branch
.macro C
.if 0
.exitm
.else
.long 3
.endif
.long 3
.endm
C
// CHECK: .long 3
// CHECK: .long 3

// .exitm is in a macro that's expanded in a conditional block.
.macro D
.long 4
.exitm
.long 4
.endm
.if 1
D
.endif
// CHECK: .long 4
// CHECK-NOT: .long 4
