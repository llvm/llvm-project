; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=corei7 | FileCheck %s
; rdar://7396984

@str = private unnamed_addr constant [28 x i8] c"xxxxxxxxxxxxxxxxxxxxxxxxxxx\00", align 1

define void @t(i32 %count) ssp nounwind {
entry:
; CHECK-LABEL: t:
; CHECK: movups L_str+12(%rip), %xmm0
; CHECK: movups L_str(%rip), %xmm1
  %tmp0 = alloca [60 x i8], align 1
  br label %bb1

bb1:
; CHECK: LBB0_1:
; CHECK: movups %xmm0, 12(%rsp)
; CHECK: movaps %xmm1, (%rsp)
  %tmp2 = phi i32 [ %tmp3, %bb1 ], [ 0, %entry ]
  call void @llvm.memcpy.p0.p0.i64(ptr %tmp0, ptr @str, i64 28, i1 false)
  %tmp3 = add i32 %tmp2, 1
  %tmp4 = icmp eq i32 %tmp3, %count
  br i1 %tmp4, label %bb2, label %bb1

bb2:
  ret void
}

declare void @llvm.memcpy.p0.p0.i64(ptr nocapture, ptr nocapture, i64, i1) nounwind
