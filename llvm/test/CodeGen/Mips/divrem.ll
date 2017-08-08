; RUN: llc -march=mips   -mcpu=mips32   -verify-machineinstrs    -relocation-model=pic < %s | FileCheck %s -check-prefixes=ALL,ACC32,ACC32-TRAP
; RUN: llc -march=mips   -mcpu=mips32r2 -verify-machineinstrs    -relocation-model=pic < %s | FileCheck %s -check-prefixes=ALL,ACC32,ACC32-TRAP
; RUN: llc -march=mips   -mcpu=mips32r6 -verify-machineinstrs    -relocation-model=pic < %s | FileCheck %s -check-prefixes=ALL,GPR32,GPR32-TRAP
; RUN: llc -march=mips64 -mcpu=mips64   -verify-machineinstrs    -relocation-model=pic < %s | FileCheck %s -check-prefixes=ALL,ACC64,ACC64-TRAP
; RUN: llc -march=mips64 -mcpu=mips64r2 -verify-machineinstrs    -relocation-model=pic < %s | FileCheck %s -check-prefixes=ALL,ACC64,ACC64-TRAP
; RUN: llc -march=mips64 -mcpu=mips64r6 -verify-machineinstrs    -relocation-model=pic < %s | FileCheck %s -check-prefixes=ALL,GPR64,GPR64-TRAP

; RUN: llc -march=mips   -mcpu=mips32   -mno-check-zero-division -relocation-model=pic < %s | FileCheck %s -check-prefixes=ALL,ACC32,NOCHECK
; RUN: llc -march=mips   -mcpu=mips32r2 -mno-check-zero-division -relocation-model=pic < %s | FileCheck %s -check-prefixes=ALL,ACC32,NOCHECK
; RUN: llc -march=mips   -mcpu=mips32r6 -mno-check-zero-division -relocation-model=pic < %s | FileCheck %s -check-prefixes=ALL,GPR32,NOCHECK
; RUN: llc -march=mips64 -mcpu=mips64   -mno-check-zero-division -relocation-model=pic < %s | FileCheck %s -check-prefixes=ALL,ACC64,NOCHECK
; RUN: llc -march=mips64 -mcpu=mips64r2 -mno-check-zero-division -relocation-model=pic < %s | FileCheck %s -check-prefixes=ALL,ACC64,NOCHECK
; RUN: llc -march=mips64 -mcpu=mips64r6 -mno-check-zero-division -relocation-model=pic < %s | FileCheck %s -check-prefixes=ALL,GPR64,NOCHECK

; FileCheck Prefixes:
;   ALL - All targets
;   ACC32 - Accumulator based multiply/divide on 32-bit targets
;   ACC64 - Same as ACC32 but only for 64-bit targets
;   GPR32 - GPR based multiply/divide on 32-bit targets
;   GPR64 - Same as GPR32 but only for 64-bit targets
;   ACC32-TRAP - Same as TRAP and ACC32 combined
;   ACC64-TRAP - Same as TRAP and ACC64 combined
;   GPR32-TRAP - Same as TRAP and GPR32 combined
;   GPR64-TRAP - Same as TRAP and GPR64 combined
;   NOCHECK - Division by zero will not be detected

@g0 = common global i32 0, align 4
@g1 = common global i32 0, align 4

define i32 @sdiv1(i32 signext %a0, i32 signext %a1) nounwind readnone {
entry:
; ALL-LABEL: sdiv1:

; ACC32:         div $zero, $4, $5
; ACC32-TRAP:    teq $5, $zero, 7

; ACC64:         div $zero, $4, $5
; ACC64-TRAP:    teq $5, $zero, 7

; GPR32:         div $2, $4, $5
; GPR32-TRAP:    teq $5, $zero, 7

; GPR64:         div $2, $4, $5
; GPR64-TRAP:    teq $5, $zero, 7

; NOCHECK-NOT:   teq

; ACC32:         mflo $2
; ACC64:         mflo $2

; ALL: .end sdiv1

  %div = sdiv i32 %a0, %a1
  ret i32 %div
}

define i32 @srem1(i32 signext %a0, i32 signext %a1) nounwind readnone {
entry:
; ALL-LABEL: srem1:

; ACC32:         div $zero, $4, $5
; ACC32-TRAP:    teq $5, $zero, 7

; ACC64:         div $zero, $4, $5
; ACC64-TRAP:    teq $5, $zero, 7

; GPR32:         mod $2, $4, $5
; GPR32-TRAP:    teq $5, $zero, 7

; GPR64:         mod $2, $4, $5
; GPR64-TRAP:    teq $5, $zero, 7

; NOCHECK-NOT:   teq

; ACC32:         mfhi $2
; ACC64:         mfhi $2

; ALL: .end srem1

  %rem = srem i32 %a0, %a1
  ret i32 %rem
}

define i32 @udiv1(i32 signext %a0, i32 signext %a1) nounwind readnone {
entry:
; ALL-LABEL: udiv1:

; ACC32:         divu $zero, $4, $5
; ACC32-TRAP:    teq $5, $zero, 7

; ACC64:         divu $zero, $4, $5
; ACC64-TRAP:    teq $5, $zero, 7

; GPR32:         divu $2, $4, $5
; GPR32-TRAP:    teq $5, $zero, 7

; GPR64:         divu $2, $4, $5
; GPR64-TRAP:    teq $5, $zero, 7

; NOCHECK-NOT:   teq

; ACC32:         mflo $2
; ACC64:         mflo $2

; ALL: .end udiv1
  %div = udiv i32 %a0, %a1
  ret i32 %div
}

define i32 @urem1(i32 signext %a0, i32 signext %a1) nounwind readnone {
entry:
; ALL-LABEL: urem1:

; ACC32:         divu $zero, $4, $5
; ACC32-TRAP:    teq $5, $zero, 7

; ACC64:         divu $zero, $4, $5
; ACC64-TRAP:    teq $5, $zero, 7

; GPR32:         modu $2, $4, $5
; GPR32-TRAP:    teq $5, $zero, 7

; GPR64:         modu $2, $4, $5
; GPR64-TRAP:    teq $5, $zero, 7

; NOCHECK-NOT:   teq

; ACC32:         mfhi $2
; ACC64:         mfhi $2

; ALL: .end urem1

  %rem = urem i32 %a0, %a1
  ret i32 %rem
}

define i32 @sdivrem1(i32 signext %a0, i32 signext %a1, i32* nocapture %r) nounwind {
entry:
; ALL-LABEL: sdivrem1:

; ACC32:         div $zero, $4, $5
; ACC32-TRAP:    teq $5, $zero, 7
; NOCHECK-NOT:   teq
; ACC32:         mflo $2
; ACC32:         mfhi $[[R0:[0-9]+]]
; ACC32:         sw $[[R0]], 0(${{[0-9]+}})

; ACC64:         div $zero, $4, $5
; ACC64-TRAP:    teq $5, $zero, 7
; NOCHECK-NOT:   teq
; ACC64:         mflo $2
; ACC64:         mfhi $[[R0:[0-9]+]]
; ACC64:         sw $[[R0]], 0(${{[0-9]+}})

; GPR32-DAG:     mod $[[R0:[0-9]+]], $4, $5
; GPR32-TRAP:    teq $5, $zero, 7
; NOCHECK-NOT:   teq
; GPR32:         sw $[[R0]], 0(${{[0-9]+}})
; GPR32-DAG:     div $2, $4, $5
; GPR32-TRAP:    teq $5, $zero, 7

; GPR64-DAG:     mod $[[R0:[0-9]+]], $4, $5
; GPR64-TRAP:    teq $5, $zero, 7
; NOCHECK-NOT:   teq
; GPR64:         sw $[[R0]], 0(${{[0-9]+}})
; GPR64-DAG:     div $2, $4, $5
; GPR64-TRAP:    teq $5, $zero, 7
; NOCHECK-NOT:   teq

; ALL: .end sdivrem1

  %rem = srem i32 %a0, %a1
  store i32 %rem, i32* %r, align 4
  %div = sdiv i32 %a0, %a1
  ret i32 %div
}

define i32 @udivrem1(i32 signext %a0, i32 signext %a1, i32* nocapture %r) nounwind {
entry:
; ALL-LABEL: udivrem1:

; ACC32:         divu $zero, $4, $5
; ACC32-TRAP:    teq $5, $zero, 7
; NOCHECK-NOT:   teq
; ACC32:         mflo $2
; ACC32:         mfhi $[[R0:[0-9]+]]
; ACC32:         sw $[[R0]], 0(${{[0-9]+}})

; ACC64:         divu $zero, $4, $5
; ACC64-TRAP:    teq $5, $zero, 7
; NOCHECK-NOT:   teq
; ACC64:         mflo $2
; ACC64:         mfhi $[[R0:[0-9]+]]
; ACC64:         sw $[[R0]], 0(${{[0-9]+}})

; GPR32-DAG:     modu $[[R0:[0-9]+]], $4, $5
; GPR32-TRAP:    teq $5, $zero, 7
; GPR32:         sw $[[R0]], 0(${{[0-9]+}})
; NOCHECK-NOT:   teq
; GPR32-DAG:     divu $2, $4, $5
; GPR32-TRAP:    teq $5, $zero, 7
; NOCHECK-NOT:   teq

; GPR64-DAG:     modu $[[R0:[0-9]+]], $4, $5
; GPR64-TRAP:    teq $5, $zero, 7
; NOCHECK-NOT:   teq
; GPR64:         sw $[[R0]], 0(${{[0-9]+}})
; GPR64-DAG:     divu $2, $4, $5
; GPR64-TRAP:    teq $5, $zero, 7
; NOCHECK-NOT:   teq

; ALL: .end udivrem1

  %rem = urem i32 %a0, %a1
  store i32 %rem, i32* %r, align 4
  %div = udiv i32 %a0, %a1
  ret i32 %div
}

; FIXME: It's not clear what this is supposed to test.
define i32 @killFlags() {
entry:
  %0 = load i32, i32* @g0, align 4
  %1 = load i32, i32* @g1, align 4
  %div = sdiv i32 %0, %1
  ret i32 %div
}

define i64 @sdiv2(i64 %a0, i64 %a1) nounwind readnone {
entry:
; ALL-LABEL: sdiv2:

; ACC32:         lw $25, %call16(__divdi3)(
; ACC32:         jalr $25

; ACC64:         ddiv $zero, $4, $5
; ACC64-TRAP:    teq $5, $zero, 7

; GPR64:         ddiv $2, $4, $5
; GPR64-TRAP:    teq $5, $zero, 7

; NOCHECK-NOT:   teq

; ACC64:         mflo $2

; ALL: .end sdiv2

  %div = sdiv i64 %a0, %a1
  ret i64 %div
}

define i64 @srem2(i64 %a0, i64 %a1) nounwind readnone {
entry:
; ALL-LABEL: srem2:

; ACC32:         lw $25, %call16(__moddi3)(
; ACC32:         jalr $25

; ACC64:         div $zero, $4, $5
; ACC64-TRAP:    teq $5, $zero, 7

; GPR64:         dmod $2, $4, $5
; GPR64-TRAP:    teq $5, $zero, 7

; NOCHECK-NOT:   teq

; ACC64:         mfhi $2

; ALL: .end srem2

  %rem = srem i64 %a0, %a1
  ret i64 %rem
}

define i64 @udiv2(i64 %a0, i64 %a1) nounwind readnone {
entry:
; ALL-LABEL: udiv2:

; ACC32:         lw $25, %call16(__udivdi3)(
; ACC32:         jalr $25

; ACC64:         divu $zero, $4, $5
; ACC64-TRAP:    teq $5, $zero, 7

; GPR64:         ddivu $2, $4, $5
; GPR64-TRAP:    teq $5, $zero, 7

; NOCHECK-NOT:   teq

; ACC64:         mflo $2

; ALL: .end udiv2
  %div = udiv i64 %a0, %a1
  ret i64 %div
}

define i64 @urem2(i64 %a0, i64 %a1) nounwind readnone {
entry:
; ALL-LABEL: urem2:

; ACC32:         lw $25, %call16(__umoddi3)(
; ACC32:         jalr $25

; ACC64:         divu $zero, $4, $5
; ACC64-TRAP:    teq $5, $zero, 7

; GPR64:         dmodu $2, $4, $5
; GPR64-TRAP:    teq $5, $zero, 7

; NOCHECK-NOT:   teq

; ACC64:         mfhi $2

; ALL: .end urem2

  %rem = urem i64 %a0, %a1
  ret i64 %rem
}

define i64 @sdivrem2(i64 %a0, i64 %a1, i64* nocapture %r) nounwind {
entry:
; ALL-LABEL: sdivrem2:

; sdivrem2 is too complex to effectively check. We can at least check for the
; calls though.
; ACC32:         lw $25, %call16(__moddi3)(
; ACC32:         jalr $25
; ACC32:         lw $25, %call16(__divdi3)(
; ACC32:         jalr $25

; ACC64:         ddiv $zero, $4, $5
; ACC64-TRAP:    teq $5, $zero, 7
; NOCHECK-NOT:   teq
; ACC64:         mflo $2
; ACC64:         mfhi $[[R0:[0-9]+]]
; ACC64:         sd $[[R0]], 0(${{[0-9]+}})

; GPR64-DAG:     dmod $[[R0:[0-9]+]], $4, $5
; GPR64-TRAP:    teq $5, $zero, 7
; NOCHECK-NOT:   teq
; GPR64:         sd $[[R0]], 0(${{[0-9]+}})

; GPR64-DAG:     ddiv $2, $4, $5
; GPR64-TRAP:    teq $5, $zero, 7
; NOCHECK-NOT:   teq

; ALL: .end sdivrem2

  %rem = srem i64 %a0, %a1
  store i64 %rem, i64* %r, align 8
  %div = sdiv i64 %a0, %a1
  ret i64 %div
}

define i64 @udivrem2(i64 %a0, i64 %a1, i64* nocapture %r) nounwind {
entry:
; ALL-LABEL: udivrem2:

; udivrem2 is too complex to effectively check. We can at least check for the
; calls though.
; ACC32:         lw $25, %call16(__umoddi3)(
; ACC32:         jalr $25
; ACC32:         lw $25, %call16(__udivdi3)(
; ACC32:         jalr $25

; ACC64:         ddivu $zero, $4, $5
; ACC64-TRAP:    teq $5, $zero, 7
; NOCHECK-NOT:   teq
; ACC64:         mflo $2
; ACC64:         mfhi $[[R0:[0-9]+]]
; ACC64:         sd $[[R0]], 0(${{[0-9]+}})

; GPR64:         dmodu $[[R0:[0-9]+]], $4, $5
; GPR64-TRAP:    teq $5, $zero, 7
; NOCHECK-NOT:   teq
; GPR64:         sd $[[R0]], 0(${{[0-9]+}})

; GPR64-DAG:     ddivu $2, $4, $5
; GPR64-TRAP:    teq $5, $zero, 7
; NOCHECK-NOT:   teq

; ALL: .end udivrem2

  %rem = urem i64 %a0, %a1
  store i64 %rem, i64* %r, align 8
  %div = udiv i64 %a0, %a1
  ret i64 %div
}
