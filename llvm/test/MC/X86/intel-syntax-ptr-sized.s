// RUN: llvm-mc %s -triple=i686-pc-windows | FileCheck %s

.intel_syntax

push [eax]
// CHECK: pushl (%eax)
call [eax]
// CHECK: calll *(%eax)
jmp [eax]
// CHECK: jmpl *(%eax)

// mode switch
.code16

push [eax]
// CHECK: pushw (%eax)
call [eax]
// CHECK: callw *(%eax)
jmp [eax]
// CHECK: jmpw *(%eax)
