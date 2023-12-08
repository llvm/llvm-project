; RUN: opt -S -loop-reduce < %s | FileCheck %s

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-linux-gnu"

; This test check SCEVExpander FactorOutConstant() is not crashing with blind cast 'Factor' to SCEVConstant.

; CHECK-LABEL: test
; FIXME: Handle VectorType in SCEVExpander::expandAddToGEP.
; The generated IR is not ideal with base 'scalar_vector' cast to i8*, and do ugly getelementptr over casted base.
; CHECK: scevgep
define void @test(ptr %a, i32 %v, i64 %n) {
entry:
  %scalar_vector = alloca <vscale x 4 x i32>, align 16
  %num_elm = call i64 @llvm.aarch64.sve.cntw(i32 31)
  %scalar_count = and i64 %num_elm, -4
  br label %loop_header

exit:
  ret void

loop_header:
  %indvar = phi i64 [ 0, %entry ], [ %indvar_next, %for_loop ]
  br label %scalar_loop

scalar_loop:
  %gep_vec = phi ptr [ %scalar_vector, %loop_header ], [ %gep_vec_inc, %scalar_loop ]
  %scalar_iv = phi i64 [ 0, %loop_header ], [ %scalar_iv_next, %scalar_loop ]
  store i32 %v, ptr %gep_vec, align 4
  %scalar_iv_next = add i64 %scalar_iv, 1
  %gep_vec_inc = getelementptr i32, ptr %gep_vec, i64 1
  %scalar_exit = icmp eq i64 %scalar_iv_next, %scalar_count
  br i1 %scalar_exit, label %for_loop, label %scalar_loop

for_loop:
  %vector = load <vscale x 4 x i32>, ptr %scalar_vector, align 16
  %gep_a = getelementptr i32, ptr %a, i64 %indvar
  call void @llvm.masked.store.nxv4i32.p0(<vscale x 4 x i32> %vector, ptr %gep_a, i32 4, <vscale x 4 x i1> undef)
  %indvar_next = add nsw i64 %indvar, %scalar_count
  %exit_cond = icmp eq i64 %indvar_next, %n
  br i1 %exit_cond, label %exit, label %loop_header
}

declare i64 @llvm.aarch64.sve.cntw(i32 immarg)

declare void @llvm.masked.store.nxv4i32.p0(<vscale x 4 x i32>, ptr, i32 immarg, <vscale x 4 x i1>)
