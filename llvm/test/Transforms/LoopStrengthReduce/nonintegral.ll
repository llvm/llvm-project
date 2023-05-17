; RUN: opt -opaque-pointers=0 -S -loop-reduce < %s | FileCheck %s

; Address Space 10 is non-integral. The optimizer is not allowed to use
; ptrtoint/inttoptr instructions. Make sure that this doesn't happen
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128-ni:10:11:12"
target triple = "x86_64-unknown-linux-gnu"

define void @japi1__unsafe_getindex_65028(i64 addrspace(10)* %arg) {
; CHECK-NOT: inttoptr
; CHECK-NOT: ptrtoint
; How exactly SCEV chooses to materialize isn't all that important, as
; long as it doesn't try to round-trip through integers. As of this writing,
; it emits a byte-wise gep, which is fine.
; CHECK: getelementptr i64, i64 addrspace(10)* {{.*}}, i64 {{.*}}
top:
  br label %L86

L86:                                              ; preds = %L86, %top
  %i.0 = phi i64 [ 0, %top ], [ %tmp, %L86 ]
  %tmp = add i64 %i.0, 1
  br i1 undef, label %L86, label %if29

if29:                                             ; preds = %L86
  %tmp1 = shl i64 %tmp, 1
  %tmp2 = add i64 %tmp1, -2
  br label %if31

if31:                                             ; preds = %if38, %if29
  %"#temp#1.sroa.0.022" = phi i64 [ 0, %if29 ], [ %tmp3, %if38 ]
  br label %L119

L119:                                             ; preds = %L119, %if31
  %i5.0 = phi i64 [ %"#temp#1.sroa.0.022", %if31 ], [ %tmp3, %L119 ]
  %tmp3 = add i64 %i5.0, 1
  br i1 undef, label %L119, label %if38

if38:                                             ; preds = %L119
  %tmp4 = add i64 %tmp2, %i5.0
  %tmp5 = getelementptr i64, i64 addrspace(10)* %arg, i64 %tmp4
  %tmp6 = load i64, i64 addrspace(10)* %tmp5
  br i1 undef, label %done, label %if31

done:                                             ; preds = %if38
  ret void
}
