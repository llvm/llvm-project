; RUN: split-file %s %t
; RUN: llc -mtriple=riscv32 -mattr=+smrnmi -verify-machineinstrs \
; RUN:   < %t/standard.ll | FileCheck %t/standard.ll
; RUN: llc -mtriple=riscv64 -mattr=+smrnmi -verify-machineinstrs \
; RUN:   < %t/standard.ll | FileCheck %t/standard.ll
; RUN: llc -mtriple=riscv32 -mattr=+experimental-xsfmclic \
; RUN:   -verify-machineinstrs < %t/sifive.ll | FileCheck %t/sifive.ll
; RUN: llc -mtriple=riscv64 -mattr=+experimental-xsfmclic \
; RUN:   -verify-machineinstrs < %t/sifive.ll | FileCheck %t/sifive.ll
; RUN: llc -mtriple=riscv32 -mattr=+xqciint -verify-machineinstrs \
; RUN:   < %t/qci.ll | FileCheck %t/qci.ll

;--- standard.ll

declare ptr @llvm.thread.pointer()
declare void @callee()

define void @machine() "interrupt"="machine" {
; CHECK-LABEL: machine:
; CHECK:       # %bb.0:
; CHECK:       beqz tp,
; CHECK:       addi sp, sp,
entry:
  %tp = call ptr @llvm.thread.pointer()
  %isnull = icmp eq ptr %tp, null
  br i1 %isnull, label %cold, label %return

return:
  ret void

cold:
  call void @callee()
  ret void
}

define void @supervisor() "interrupt"="supervisor" {
; CHECK-LABEL: supervisor:
; CHECK:       # %bb.0:
; CHECK:       beqz tp,
; CHECK:       addi sp, sp,
entry:
  %tp = call ptr @llvm.thread.pointer()
  %isnull = icmp eq ptr %tp, null
  br i1 %isnull, label %cold, label %return

return:
  ret void

cold:
  call void @callee()
  ret void
}

define void @rnmi() "interrupt"="rnmi" {
; CHECK-LABEL: rnmi:
; CHECK:       # %bb.0:
; CHECK:       beqz tp,
; CHECK:       addi sp, sp,
entry:
  %tp = call ptr @llvm.thread.pointer()
  %isnull = icmp eq ptr %tp, null
  br i1 %isnull, label %cold, label %return

return:
  ret void

cold:
  call void @callee()
  ret void
}

;--- sifive.ll

declare ptr @llvm.thread.pointer()
declare void @callee()

define void @sifive_stack_swap() "interrupt"="SiFive-CLIC-stack-swap" {
; CHECK-LABEL: sifive_stack_swap:
; CHECK:       # %bb.0:
; CHECK-NEXT:  csrrw sp, sf.mscratchcsw, sp
; CHECK-NEXT:  addi sp, sp,
; CHECK:       bnez tp,
entry:
  %tp = call ptr @llvm.thread.pointer()
  %isnull = icmp eq ptr %tp, null
  br i1 %isnull, label %cold, label %return

return:
  ret void

cold:
  call void @callee()
  ret void
}

define void @sifive_preemptible() "interrupt"="SiFive-CLIC-preemptible" {
; CHECK-LABEL: sifive_preemptible:
; CHECK:       # %bb.0:
; CHECK:       csrsi mstatus, 8
; CHECK:       bnez tp,
entry:
  %tp = call ptr @llvm.thread.pointer()
  %isnull = icmp eq ptr %tp, null
  br i1 %isnull, label %cold, label %return

return:
  ret void

cold:
  call void @callee()
  ret void
}

define void @sifive_preemptible_stack_swap() "interrupt"="SiFive-CLIC-preemptible-stack-swap" {
; CHECK-LABEL: sifive_preemptible_stack_swap:
; CHECK:       # %bb.0:
; CHECK-NEXT:  csrrw sp, sf.mscratchcsw, sp
; CHECK:       csrsi mstatus, 8
; CHECK:       bnez tp,
entry:
  %tp = call ptr @llvm.thread.pointer()
  %isnull = icmp eq ptr %tp, null
  br i1 %isnull, label %cold, label %return

return:
  ret void

cold:
  call void @callee()
  ret void
}

;--- qci.ll

declare ptr @llvm.thread.pointer()
declare void @panic() noreturn
declare void @llvm.trap() cold noreturn nounwind

define void @qci_nest() noreturn "interrupt"="qci-nest" {
; CHECK-LABEL: qci_nest:
; CHECK:       # %bb.0:
; CHECK-NEXT:  qc.c.mienter.nest
; CHECK:       bnez tp,
entry:
  %tp = call ptr @llvm.thread.pointer()
  %isnull = icmp eq ptr %tp, null
  br i1 %isnull, label %panic, label %trap

trap:
  call void @llvm.trap()
  unreachable

panic:
  call void @panic()
  unreachable
}

define void @qci_nonest() noreturn "interrupt"="qci-nonest" {
; CHECK-LABEL: qci_nonest:
; CHECK:       # %bb.0:
; CHECK-NEXT:  qc.c.mienter
; CHECK:       bnez tp,
entry:
  %tp = call ptr @llvm.thread.pointer()
  %isnull = icmp eq ptr %tp, null
  br i1 %isnull, label %panic, label %trap

trap:
  call void @llvm.trap()
  unreachable

panic:
  call void @panic()
  unreachable
}
