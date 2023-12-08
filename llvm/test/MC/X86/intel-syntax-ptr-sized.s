// RUN: llvm-mc %s -triple=i686-pc-windows | FileCheck %s

.intel_syntax

push [eax]
// CHECK: pushl (%eax)
call [eax]
// CHECK: calll *(%eax)
jmp [eax]
// CHECK: jmpl *(%eax)

lgdt [eax]
// CHECK: lgdtl (%eax)
lidt [eax]
// CHECK: lidtl (%eax)
sgdt [eax]
// CHECK: sgdtl (%eax)
sidt [eax]
// CHECK: sidtl (%eax)

// mode switch
.code16

push [eax]
// CHECK: pushw (%eax)
call [eax]
// CHECK: callw *(%eax)
jmp [eax]
// CHECK: jmpw *(%eax)

lgdt [eax]
// CHECK: lgdtw (%eax)
lidt [eax]
// CHECK: lidtw (%eax)
sgdt [eax]
// CHECK: sgdtw (%eax)
sidt [eax]
// CHECK: sidtw (%eax)
