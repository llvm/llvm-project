; RUN: llc -verify-machineinstrs -mcpu=x86-64-v3 -mtriple=x86_64-unknown-unknown < %s | FileCheck %s --check-prefixes=FAST,ALL
; RUN: llc -verify-machineinstrs -mcpu=x86-64-v3 -mattr=+false-deps-bls -mtriple=x86_64-unknown-unknown < %s | FileCheck %s --check-prefixes=SLOW,ALL
; RUN: llc -verify-machineinstrs -mcpu=haswell -mtriple=x86_64-unknown-unknown < %s | FileCheck %s --check-prefixes=FAST,ALL
; RUN: llc -verify-machineinstrs -mcpu=skylake -mtriple=x86_64-unknown-unknown < %s | FileCheck %s --check-prefixes=FAST,ALL
; RUN: llc -verify-machineinstrs -mcpu=skx -mtriple=x86_64-unknown-unknown < %s | FileCheck %s --check-prefixes=FAST,ALL
; RUN: llc -verify-machineinstrs -mcpu=icelake-client -mtriple=x86_64-unknown-unknown < %s | FileCheck %s --check-prefixes=FAST,ALL
; RUN: llc -verify-machineinstrs -mcpu=alderlake -mtriple=x86_64-unknown-unknown < %s | FileCheck %s --check-prefixes=FAST,ALL
; RUN: llc -verify-machineinstrs -mcpu=sapphirerapids -mtriple=x86_64-unknown-unknown < %s | FileCheck %s --check-prefixes=FAST,ALL
; RUN: llc -verify-machineinstrs -mcpu=emeraldrapids -mtriple=x86_64-unknown-unknown < %s | FileCheck %s --check-prefixes=FAST,ALL
; RUN: llc -verify-machineinstrs -mcpu=graniterapids -mtriple=x86_64-unknown-unknown < %s | FileCheck %s --check-prefixes=FAST,ALL
; RUN: llc -verify-machineinstrs -mcpu=znver4 -mtriple=x86_64-unknown-unknown < %s | FileCheck %s --check-prefixes=FAST,ALL
; RUN: llc -verify-machineinstrs -mcpu=znver5 -mtriple=x86_64-unknown-unknown < %s | FileCheck %s --check-prefixes=SLOW,ALL
; RUN: llc -verify-machineinstrs -mcpu=znver6 -mtriple=x86_64-unknown-unknown < %s | FileCheck %s --check-prefixes=FAST,ALL


; Test that dependencies on the output register are broken if it was written recently (write simulated using inline asm).


define i32 @blsi32_rr(i32 %a0) {
; SLOW-LABEL: blsi32_rr:
; SLOW:       # %bb.0:
; SLOW-NEXT:    #APP
; SLOW-NEXT:    #NO_APP
; SLOW-NEXT:    xorl %eax, %eax
; SLOW-NEXT:    blsil %edi, %eax
; SLOW-NEXT:    retq
;
; FAST-LABEL: blsi32_rr:
; FAST:       # %bb.0:
; FAST-NEXT:    #APP
; FAST-NEXT:    #NO_APP
; FAST-NEXT:    blsil %edi, %eax
; FAST-NEXT:    retq
  %1 = tail call <2 x i64> asm sideeffect "", "=x,~{eax}"()
  %2 = sub i32 0, %a0
  %3 = and i32 %2, %a0
  ret i32 %3
}

define i64 @blsi64_rr(i64 %a0) {
; SLOW-LABEL: blsi64_rr:
; SLOW:       # %bb.0:
; SLOW-NEXT:    #APP
; SLOW-NEXT:    #NO_APP
; SLOW-NEXT:    xorl %eax, %eax
; SLOW-NEXT:    blsiq %rdi, %rax
; SLOW-NEXT:    retq
;
; FAST-LABEL: blsi64_rr:
; FAST:       # %bb.0:
; FAST-NEXT:    #APP
; FAST-NEXT:    #NO_APP
; FAST-NEXT:    blsiq %rdi, %rax
; FAST-NEXT:    retq
  %1 = tail call <2 x i64> asm sideeffect "", "=x,~{eax}"()
  %2 = sub i64 0, %a0
  %3 = and i64 %2, %a0
  ret i64 %3
}

define i32 @blsi32_rm(ptr %p0) {
; SLOW-LABEL: blsi32_rm:
; SLOW:       # %bb.0:
; SLOW-NEXT:    #APP
; SLOW-NEXT:    #NO_APP
; SLOW-NEXT:    xorl %eax, %eax
; SLOW-NEXT:    blsil (%rdi), %eax
; SLOW-NEXT:    retq
;
; FAST-LABEL: blsi32_rm:
; FAST:       # %bb.0:
; FAST-NEXT:    #APP
; FAST-NEXT:    #NO_APP
; FAST-NEXT:    blsil (%rdi), %eax
; FAST-NEXT:    retq
  %1 = tail call <2 x i64> asm sideeffect "", "=x,~{eax}"()
  %a0 = load i32, ptr %p0, align 4
  %2 = sub i32 0, %a0
  %3 = and i32 %2, %a0
  ret i32 %3
}

define i64 @blsi64_rm(ptr %p0) {
; SLOW-LABEL: blsi64_rm:
; SLOW:       # %bb.0:
; SLOW-NEXT:    #APP
; SLOW-NEXT:    #NO_APP
; SLOW-NEXT:    xorl %eax, %eax
; SLOW-NEXT:    blsiq (%rdi), %rax
; SLOW-NEXT:    retq
;
; FAST-LABEL: blsi64_rm:
; FAST:       # %bb.0:
; FAST-NEXT:    #APP
; FAST-NEXT:    #NO_APP
; FAST-NEXT:    blsiq (%rdi), %rax
; FAST-NEXT:    retq
  %1 = tail call <2 x i64> asm sideeffect "", "=x,~{eax}"()
  %a0 = load i64, ptr %p0, align 4
  %2 = sub i64 0, %a0
  %3 = and i64 %2, %a0
  ret i64 %3
}



define i32 @blsr32_rr(i32 %a0) {
; SLOW-LABEL: blsr32_rr:
; SLOW:       # %bb.0:
; SLOW-NEXT:    #APP
; SLOW-NEXT:    #NO_APP
; SLOW-NEXT:    xorl %eax, %eax
; SLOW-NEXT:    blsrl %edi, %eax
; SLOW-NEXT:    retq
;
; FAST-LABEL: blsr32_rr:
; FAST:       # %bb.0:
; FAST-NEXT:    #APP
; FAST-NEXT:    #NO_APP
; FAST-NEXT:    blsrl %edi, %eax
; FAST-NEXT:    retq
  %1 = tail call <2 x i64> asm sideeffect "", "=x,~{eax}"()
  %2 = add i32 %a0, -1
  %3 = and i32 %a0, %2
  ret i32 %3
}

define i64 @blsr64_rr(i64 %a0) {
; SLOW-LABEL: blsr64_rr:
; SLOW:       # %bb.0:
; SLOW-NEXT:    #APP
; SLOW-NEXT:    #NO_APP
; SLOW-NEXT:    xorl %eax, %eax
; SLOW-NEXT:    blsrq %rdi, %rax
; SLOW-NEXT:    retq
;
; FAST-LABEL: blsr64_rr:
; FAST:       # %bb.0:
; FAST-NEXT:    #APP
; FAST-NEXT:    #NO_APP
; FAST-NEXT:    blsrq %rdi, %rax
; FAST-NEXT:    retq
  %1 = tail call <2 x i64> asm sideeffect "", "=x,~{eax}"()
  %2 = add i64 %a0, -1
  %3 = and i64 %a0, %2
  ret i64 %3
}

define i32 @blsr32_rm(ptr %p0) {
; SLOW-LABEL: blsr32_rm:
; SLOW:       # %bb.0:
; SLOW-NEXT:    #APP
; SLOW-NEXT:    #NO_APP
; SLOW-NEXT:    xorl %eax, %eax
; SLOW-NEXT:    blsrl (%rdi), %eax
; SLOW-NEXT:    retq
;
; FAST-LABEL: blsr32_rm:
; FAST:       # %bb.0:
; FAST-NEXT:    #APP
; FAST-NEXT:    #NO_APP
; FAST-NEXT:    blsrl (%rdi), %eax
; FAST-NEXT:    retq
  %1 = tail call <2 x i64> asm sideeffect "", "=x,~{eax}"()
  %a0 = load i32, ptr %p0, align 4
  %2 = add i32 %a0, -1
  %3 = and i32 %a0, %2
  ret i32 %3
}

define i64 @blsr64_rm(ptr %p0) {
; SLOW-LABEL: blsr64_rm:
; SLOW:       # %bb.0:
; SLOW-NEXT:    #APP
; SLOW-NEXT:    #NO_APP
; SLOW-NEXT:    xorl %eax, %eax
; SLOW-NEXT:    blsrq (%rdi), %rax
; SLOW-NEXT:    retq
;
; FAST-LABEL: blsr64_rm:
; FAST:       # %bb.0:
; FAST-NEXT:    #APP
; FAST-NEXT:    #NO_APP
; FAST-NEXT:    blsrq (%rdi), %rax
; FAST-NEXT:    retq
  %1 = tail call <2 x i64> asm sideeffect "", "=x,~{eax}"()
  %a0 = load i64, ptr %p0, align 4
  %2 = add i64 %a0, -1
  %3 = and i64 %a0, %2
  ret i64 %3
}



define i32 @blsmsk32_rr(i32 %a0) {
; SLOW-LABEL: blsmsk32_rr:
; SLOW:       # %bb.0:
; SLOW-NEXT:    #APP
; SLOW-NEXT:    #NO_APP
; SLOW-NEXT:    xorl %eax, %eax
; SLOW-NEXT:    blsmskl %edi, %eax
; SLOW-NEXT:    retq
;
; FAST-LABEL: blsmsk32_rr:
; FAST:       # %bb.0:
; FAST-NEXT:    #APP
; FAST-NEXT:    #NO_APP
; FAST-NEXT:    blsmskl %edi, %eax
; FAST-NEXT:    retq
  %1 = tail call <2 x i64> asm sideeffect "", "=x,~{eax}"()
  %2 = add i32 %a0, -1
  %3 = xor i32 %a0, %2
  ret i32 %3
}

define i64 @blsmsk64_rr(i64 %a0) {
; SLOW-LABEL: blsmsk64_rr:
; SLOW:       # %bb.0:
; SLOW-NEXT:    #APP
; SLOW-NEXT:    #NO_APP
; SLOW-NEXT:    xorl %eax, %eax
; SLOW-NEXT:    blsmskq %rdi, %rax
; SLOW-NEXT:    retq
;
; FAST-LABEL: blsmsk64_rr:
; FAST:       # %bb.0:
; FAST-NEXT:    #APP
; FAST-NEXT:    #NO_APP
; FAST-NEXT:    blsmskq %rdi, %rax
; FAST-NEXT:    retq
  %1 = tail call <2 x i64> asm sideeffect "", "=x,~{eax}"()
  %2 = add i64 %a0, -1
  %3 = xor i64 %a0, %2
  ret i64 %3
}

define i32 @blsmsk32_rm(ptr %p0) {
; SLOW-LABEL: blsmsk32_rm:
; SLOW:       # %bb.0:
; SLOW-NEXT:    #APP
; SLOW-NEXT:    #NO_APP
; SLOW-NEXT:    xorl %eax, %eax
; SLOW-NEXT:    blsmskl (%rdi), %eax
; SLOW-NEXT:    retq
;
; FAST-LABEL: blsmsk32_rm:
; FAST:       # %bb.0:
; FAST-NEXT:    #APP
; FAST-NEXT:    #NO_APP
; FAST-NEXT:    blsmskl (%rdi), %eax
; FAST-NEXT:    retq
  %1 = tail call <2 x i64> asm sideeffect "", "=x,~{eax}"()
  %a0 = load i32, ptr %p0, align 4
  %2 = add i32 %a0, -1
  %3 = xor i32 %a0, %2
  ret i32 %3
}

define i64 @blsmsk64_rm(ptr %p0) {
; SLOW-LABEL: blsmsk64_rm:
; SLOW:       # %bb.0:
; SLOW-NEXT:    #APP
; SLOW-NEXT:    #NO_APP
; SLOW-NEXT:    xorl %eax, %eax
; SLOW-NEXT:    blsmskq (%rdi), %rax
; SLOW-NEXT:    retq
;
; FAST-LABEL: blsmsk64_rm:
; FAST:       # %bb.0:
; FAST-NEXT:    #APP
; FAST-NEXT:    #NO_APP
; FAST-NEXT:    blsmskq (%rdi), %rax
; FAST-NEXT:    retq
  %1 = tail call <2 x i64> asm sideeffect "", "=x,~{eax}"()
  %a0 = load i64, ptr %p0, align 4
  %2 = add i64 %a0, -1
  %3 = xor i64 %a0, %2
  ret i64 %3
}


; Test that dependencies on the output register are not broken if it is the same as the input register or it was not written recently


define i32 @no_break_blsi32_rr(i32 %a0) {
; ALL-LABEL: no_break_blsi32_rr:
; ALL:       # %bb.0:
; ALL-NEXT:    blsil %edi, %eax
; ALL-NEXT:    #APP
; ALL-NEXT:    #NO_APP
; ALL-NEXT:    blsil %eax, %eax
; ALL-NEXT:    retq
  %1 = sub i32 0, %a0
  %2 = and i32 %1, %a0
  %3 = tail call i32 asm sideeffect "", "={eax},{eax}"(i32 %2)
  %4 = sub i32 0, %3
  %5 = and i32 %4, %3
  ret i32 %5
}

define i64 @no_break_blsi64_rr(i64 %a0) {
; ALL-LABEL: no_break_blsi64_rr:
; ALL:       # %bb.0:
; ALL-NEXT:    blsiq %rdi, %rax
; ALL-NEXT:    #APP
; ALL-NEXT:    #NO_APP
; ALL-NEXT:    blsiq %rax, %rax
; ALL-NEXT:    retq
  %1 = sub i64 0, %a0
  %2 = and i64 %1, %a0
  %3 = tail call i64 asm sideeffect "", "={rax},{rax}"(i64 %2)
  %4 = sub i64 0, %3
  %5 = and i64 %4, %3
  ret i64 %5
}

define i32 @no_break_blsi32_rm(ptr %p0) {
; ALL-LABEL: no_break_blsi32_rm:
; ALL:       # %bb.0:
; ALL-NEXT:    blsil (%rdi), %eax
; ALL-NEXT:    retq
  %a0 = load i32, ptr %p0, align 4
  %1 = sub i32 0, %a0
  %2 = and i32 %1, %a0
  ret i32 %2
}

define i64 @no_break_blsi64_rm(ptr %p0) {
; ALL-LABEL: no_break_blsi64_rm:
; ALL:       # %bb.0:
; ALL-NEXT:    blsiq (%rdi), %rax
; ALL-NEXT:    retq
  %a0 = load i64, ptr %p0, align 4
  %1 = sub i64 0, %a0
  %2 = and i64 %1, %a0
  ret i64 %2
}


define i32 @no_break_blsr32_rr(i32 %a0) {
; ALL-LABEL: no_break_blsr32_rr:
; ALL:       # %bb.0:
; ALL-NEXT:    blsrl %edi, %eax
; ALL-NEXT:    #APP
; ALL-NEXT:    #NO_APP
; ALL-NEXT:    blsrl %eax, %eax
; ALL-NEXT:    retq
  %1 = add i32 %a0, -1
  %2 = and i32 %a0, %1
  %3 = tail call i32 asm sideeffect "", "={eax},{eax}"(i32 %2)
  %4 = add i32 %3, -1
  %5 = and i32 %3, %4
  ret i32 %5
}

define i64 @no_break_blsr64_rr(i64 %a0) {
; ALL-LABEL: no_break_blsr64_rr:
; ALL:       # %bb.0:
; ALL-NEXT:    blsrq %rdi, %rax
; ALL-NEXT:    #APP
; ALL-NEXT:    #NO_APP
; ALL-NEXT:    blsrq %rax, %rax
; ALL-NEXT:    retq
  %1 = add i64 %a0, -1
  %2 = and i64 %a0, %1
  %3 = tail call i64 asm sideeffect "", "={rax},{rax}"(i64 %2)
  %4 = add i64 %3, -1
  %5 = and i64 %3, %4
  ret i64 %5
}

define i32 @no_break_blsr32_rm(ptr %p0) {
; ALL-LABEL: no_break_blsr32_rm:
; ALL:       # %bb.0:
; ALL-NEXT:    blsrl (%rdi), %eax
; ALL-NEXT:    retq
  %a0 = load i32, ptr %p0, align 4
  %1 = add i32 %a0, -1
  %2 = and i32 %a0, %1
  ret i32 %2
}

define i64 @no_break_blsr64_rm(ptr %p0) {
; ALL-LABEL: no_break_blsr64_rm:
; ALL:       # %bb.0:
; ALL-NEXT:    blsrq (%rdi), %rax
; ALL-NEXT:    retq
  %a0 = load i64, ptr %p0, align 4
  %1 = add i64 %a0, -1
  %2 = and i64 %a0, %1
  ret i64 %2
}



define i32 @no_break_blsmsk32_rr(i32 %a0) {
; ALL-LABEL: no_break_blsmsk32_rr:
; ALL:       # %bb.0:
; ALL-NEXT:    blsmskl %edi, %eax
; ALL-NEXT:    #APP
; ALL-NEXT:    #NO_APP
; ALL-NEXT:    blsmskl %eax, %eax
; ALL-NEXT:    retq
  %1 = add i32 %a0, -1
  %2 = xor i32 %a0, %1
  %3 = tail call i32 asm sideeffect "", "={eax},{eax}"(i32 %2)
  %4 = add i32 %3, -1
  %5 = xor i32 %3, %4
  ret i32 %5
}

define i64 @no_break_blsmsk64_rr(i64 %a0) {
; ALL-LABEL: no_break_blsmsk64_rr:
; ALL:       # %bb.0:
; ALL-NEXT:    blsmskq %rdi, %rax
; ALL-NEXT:    #APP
; ALL-NEXT:    #NO_APP
; ALL-NEXT:    blsmskq %rax, %rax
; ALL-NEXT:    retq
  %1 = add i64 %a0, -1
  %2 = xor i64 %a0, %1
  %3 = tail call i64 asm sideeffect "", "={rax},{rax}"(i64 %2)
  %4 = add i64 %3, -1
  %5 = xor i64 %3, %4
  ret i64 %5
}

define i32 @no_break_blsmsk32_rm(ptr %p0) {
; ALL-LABEL: no_break_blsmsk32_rm:
; ALL:       # %bb.0:
; ALL-NEXT:    blsmskl (%rdi), %eax
; ALL-NEXT:    retq
  %a0 = load i32, ptr %p0, align 4
  %1 = add i32 %a0, -1
  %2 = xor i32 %a0, %1
  ret i32 %2
}

define i64 @no_break_blsmsk64_rm(ptr %p0) {
; ALL-LABEL: no_break_blsmsk64_rm:
; ALL:       # %bb.0:
; ALL-NEXT:    blsmskq (%rdi), %rax
; ALL-NEXT:    retq
  %a0 = load i64, ptr %p0, align 4
  %1 = add i64 %a0, -1
  %2 = xor i64 %a0, %1
  ret i64 %2
}
