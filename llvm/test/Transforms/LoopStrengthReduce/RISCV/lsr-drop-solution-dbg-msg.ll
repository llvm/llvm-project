; REQUIRES: asserts
; RUN: llc < %s -O3 -mattr=+v -debug -lsr-drop-solution 2>&1 | FileCheck --check-prefix=DEBUG %s
; RUN: llc < %s -O3 -mattr=+v -debug 2>&1 | FileCheck --check-prefix=DEBUG2 %s

target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n64-S128"
target triple = "riscv64-unknown-linux-gnu"

define ptr @foo(ptr %a0, ptr %a1, i64 %a2) {
;DEBUG: The chosen solution requires 3 instructions 6 regs, with addrec cost 1, plus 2 base adds, plus 5 setup cost
;DEBUG: The baseline solution requires 2 instructions 4 regs, with addrec cost 2, plus 3 setup cost
;DEBUG: Baseline is more profitable than chosen solution, dropping LSR solution.

;DEBUG2: Baseline is more profitable than chosen solution, add option 'lsr-drop-solution' to drop LSR solution.
entry:
  %0 = ptrtoint ptr %a0 to i64
  %1 = tail call i64 @llvm.riscv.vsetvli.i64(i64 %a2, i64 0, i64 3)
  %cmp.not = icmp eq i64 %1, %a2
  br i1 %cmp.not, label %if.end, label %if.then

if.then:                                        ; preds = %entry
  %add = add i64 %0, %a2
  %sub = sub i64 %add, %1
  br label %do.body

do.body:                                        ; preds = %do.body, %if.then
  %a3.0 = phi i64 [ %0, %if.then ], [ %add1, %do.body ]
  %a1.addr.0 = phi ptr [ %a1, %if.then ], [ %add.ptr, %do.body ]
  %2 = tail call <vscale x 64 x i8> @llvm.riscv.vle.nxv64i8.i64(<vscale x 64 x i8> undef, ptr %a1.addr.0, i64 %1)
  %3 = inttoptr i64 %a3.0 to ptr
  tail call void @llvm.riscv.vse.nxv64i8.i64(<vscale x 64 x i8> %2, ptr %3, i64 %1)
  %add1 = add i64 %a3.0, %1
  %add.ptr = getelementptr i8, ptr %a1.addr.0, i64 %1
  %cmp2 = icmp ugt i64 %sub, %add1
  br i1 %cmp2, label %do.body, label %do.end

do.end:                                         ; preds = %do.body
  %sub4 = sub i64 %add, %add1
  %4 = tail call i64 @llvm.riscv.vsetvli.i64(i64 %sub4, i64 0, i64 3)
  br label %if.end

if.end:                                         ; preds = %do.end, %entry
  %a3.1 = phi i64 [ %add1, %do.end ], [ %0, %entry ]
  %t0.0 = phi i64 [ %4, %do.end ], [ %a2, %entry ]
  %a1.addr.1 = phi ptr [ %add.ptr, %do.end ], [ %a1, %entry ]
  %5 = tail call <vscale x 64 x i8> @llvm.riscv.vle.nxv64i8.i64(<vscale x 64 x i8> undef, ptr %a1.addr.1, i64 %t0.0)
  %6 = inttoptr i64 %a3.1 to ptr
  tail call void @llvm.riscv.vse.nxv64i8.i64(<vscale x 64 x i8> %5, ptr %6, i64 %t0.0)
  ret ptr %a0
}

declare i64 @llvm.riscv.vsetvli.i64(i64, i64 immarg, i64 immarg)

declare <vscale x 64 x i8> @llvm.riscv.vle.nxv64i8.i64(<vscale x 64 x i8>, ptr nocapture, i64)

declare void @llvm.riscv.vse.nxv64i8.i64(<vscale x 64 x i8>, ptr nocapture, i64)
