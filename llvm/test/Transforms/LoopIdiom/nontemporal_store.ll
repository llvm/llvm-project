; RUN: opt -passes=loop-idiom < %s -S | FileCheck %s
; RUN: opt -aa-pipeline=basic-aa -passes='require<aa>,require<targetir>,require<scalar-evolution>,require<opt-remark-emit>,loop(loop-idiom)' < %s -S | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.bigBlock_t = type { [256 x <4 x float>] }

; CHECK-LABEL: @test(
; CHECK-NOT: llvm.memset
define void @test(ptr %p) {
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %index.02 = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %dst.01 = phi ptr [ %p, %entry ], [ %add.ptr2, %for.body ]
  store <4 x float> zeroinitializer, ptr %dst.01, align 16, !nontemporal !0
  %add.ptr1 = getelementptr inbounds float, ptr %dst.01, i64 4
  store <4 x float> zeroinitializer, ptr %add.ptr1, align 16, !nontemporal !0
  %add.ptr2 = getelementptr inbounds float, ptr %dst.01, i64 8
  %add = add nuw nsw i32 %index.02, 32
  %cmp = icmp ult i32 %add, 4096
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body
  ret void
}

!0 = !{i32 1}
