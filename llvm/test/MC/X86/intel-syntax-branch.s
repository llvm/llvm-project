// RUN: llvm-mc -triple i686-unknown-unknown -x86-asm-syntax=intel %s | FileCheck %s --check-prefixes=CHECK-32,CHECK
// RUN: llvm-mc -triple x86_64-unknown-unknown --defsym X64=1 -x86-asm-syntax=intel %s | FileCheck %s --check-prefixes=CHECK-64,CHECK

// RUN: not llvm-mc -triple i686-unknown-unknown --defsym ERR=1 -x86-asm-syntax=intel %s 2>&1 | FileCheck %s --check-prefixes=ERR-32

t0:
call direct_branch
jmp direct_branch
// CHECK-LABEL: t0:
// CHECK-64: callq direct_branch
// CHECK-32: calll direct_branch
// CHECK:    jmp direct_branch

t1:
call [fn_ref]
jmp [fn_ref]
// CHECK-LABEL: t1:
// CHECK-64: callq *fn_ref
// CHECK-64: jmpq *fn_ref
// CHECK-32: calll *fn_ref
// CHECK-32: jmpl *fn_ref

.ifdef X64

  t2:
  call qword ptr [fn_ref]
  jmp qword ptr [fn_ref]
  // CHECK-64-LABEL: t2:
  // CHECK-64: callq *fn_ref
  // CHECK-64: jmpq *fn_ref

  t3:
  call qword ptr [rip + fn_ref]
  jmp qword ptr [rip + fn_ref]
  // CHECK-64-LABEL: t3:
  // CHECK-64: callq *fn_ref(%rip)
  // CHECK-64: jmpq *fn_ref(%rip)

.else

  t4:
  call dword ptr [fn_ref]
  jmp dword ptr [fn_ref]
  // CHECK-32-LABEL: t4:
  // CHECK-32: calll *fn_ref
  // CHECK-32: jmpl *fn_ref

  t5:
  call dword ptr fn_ref
  jmp dword ptr fn_ref
  // CHECK-32-LABEL: t5:
  // CHECK-32: calll *fn_ref
  // CHECK-32: jmpl *fn_ref

  t6:
  call dword ptr [offset fn_ref]
  jmp dword ptr [offset fn_ref]
  // CHECK-32-LABEL: t6:
  // CHECK-32: calll *fn_ref
  // CHECK-32: jmpl *fn_ref

.ifdef ERR

  call [offset fn_ref]
  // ERR-32: {{.*}}.s:[[#@LINE-1]]:8: error: `OFFSET` operator cannot be used in an unconditional branch
  jmp [offset fn_ref]
  // ERR-32: {{.*}}.s:[[#@LINE-1]]:7: error: `OFFSET` operator cannot be used in an unconditional branch

  call offset fn_ref
  // ERR-32: {{.*}}.s:[[#@LINE-1]]:3: error: invalid operand for instruction
  jmp offset fn_ref
  // ERR-32: {{.*}}.s:[[#@LINE-1]]:3: error: invalid operand for instruction

.endif

.endif
