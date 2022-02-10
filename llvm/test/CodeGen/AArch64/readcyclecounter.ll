; RUN: llc -mtriple=aarch64-unknown-unknown -mattr=+perfmon -asm-verbose=false < %s |\
; RUN:   FileCheck %s --check-prefix=CHECK --check-prefix=PERFMON
; RUN: llc -mtriple=aarch64-unknown-unknown -mattr=-perfmon -asm-verbose=false < %s |\
; RUN:   FileCheck %s --check-prefix=CHECK --check-prefix=NOPERFMON
; RUN: llc -mtriple=aarch64-unknown-unknown -mcpu=cortex-a53 -asm-verbose=false < %s |\
; RUN:   FileCheck %s --check-prefix=CHECK --check-prefix=PERFMON
; RUN: llc -mtriple=aarch64-unknown-unknown -mcpu=cortex-a55 -asm-verbose=false < %s |\
; RUN:   FileCheck %s --check-prefix=CHECK --check-prefix=PERFMON
; RUN: llc -mtriple=aarch64-unknown-unknown -mcpu=cortex-a510 -asm-verbose=false < %s |\
; RUN:   FileCheck %s --check-prefix=CHECK --check-prefix=PERFMON
; RUN: llc -mtriple=aarch64-unknown-unknown -mcpu=cortex-a65 -asm-verbose=false < %s |\
; RUN:   FileCheck %s --check-prefix=CHECK --check-prefix=PERFMON
; RUN: llc -mtriple=aarch64-unknown-unknown -mcpu=cortex-a76 -asm-verbose=false < %s |\
; RUN:   FileCheck %s --check-prefix=CHECK --check-prefix=PERFMON
; RUN: llc -mtriple=aarch64-unknown-unknown -mcpu=cortex-a77 -asm-verbose=false < %s |\
; RUN:   FileCheck %s --check-prefix=CHECK --check-prefix=PERFMON
; RUN: llc -mtriple=aarch64-unknown-unknown -mcpu=cortex-a78 -asm-verbose=false < %s |\
; RUN:   FileCheck %s --check-prefix=CHECK --check-prefix=PERFMON
; RUN: llc -mtriple=aarch64-unknown-unknown -mcpu=cortex-a78c -asm-verbose=false < %s |\
; RUN:   FileCheck %s --check-prefix=CHECK --check-prefix=PERFMON
; RUN: llc -mtriple=aarch64-unknown-unknown -mcpu=cortex-a710 -asm-verbose=false < %s |\
; RUN:   FileCheck %s --check-prefix=CHECK --check-prefix=PERFMON
; RUN: llc -mtriple=aarch64-unknown-unknown -mcpu=cortex-r82 -asm-verbose=false < %s |\
; RUN:   FileCheck %s --check-prefix=CHECK --check-prefix=PERFMON
; RUN: llc -mtriple=aarch64-unknown-unknown -mcpu=cortex-x1 -asm-verbose=false < %s |\
; RUN:   FileCheck %s --check-prefix=CHECK --check-prefix=PERFMON
; RUN: llc -mtriple=aarch64-unknown-unknown -mcpu=cortex-x2 -asm-verbose=false < %s |\
; RUN:   FileCheck %s --check-prefix=CHECK --check-prefix=PERFMON
; RUN: llc -mtriple=aarch64-unknown-unknown -mcpu=neoverse-e1 -asm-verbose=false < %s |\
; RUN:   FileCheck %s --check-prefix=CHECK --check-prefix=PERFMON
; RUN: llc -mtriple=aarch64-unknown-unknown -mcpu=neoverse-n1 -asm-verbose=false < %s |\
; RUN:   FileCheck %s --check-prefix=CHECK --check-prefix=PERFMON
; RUN: llc -mtriple=aarch64-unknown-unknown -mcpu=neoverse-n2 -asm-verbose=false < %s |\
; RUN:   FileCheck %s --check-prefix=CHECK --check-prefix=PERFMON
; RUN: llc -mtriple=aarch64-unknown-unknown -mcpu=neoverse-v1 -asm-verbose=false < %s |\
; RUN:   FileCheck %s --check-prefix=CHECK --check-prefix=PERFMON

define i64 @test_readcyclecounter() nounwind {
  ; CHECK-LABEL:   test_readcyclecounter:
  ; PERFMON-NEXT:   mrs x0, PMCCNTR_EL0
  ; NOPERFMON-NEXT: mov x0, xzr
  ; CHECK-NEXT:     ret
  %tmp0 = call i64 @llvm.readcyclecounter()
  ret i64 %tmp0
}

declare i64 @llvm.readcyclecounter()
