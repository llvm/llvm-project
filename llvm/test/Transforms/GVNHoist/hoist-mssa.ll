; RUN: opt -S -passes='function(gvn-hoist),function(newgvn)' < %s | FileCheck %s

; Check that store hoisting works: there should be only one store left.
; CHECK-LABEL: @getopt
; CHECK: store i32
; CHECK-NOT: store i32

@optind = external global i32, align 4

define void @getopt() {
bb:
  br label %bb1

bb1:                                              ; preds = %bb
  br i1 undef, label %bb2, label %bb3

bb2:                                              ; preds = %bb1
  br label %bb13

bb3:                                              ; preds = %bb1
  br i1 undef, label %bb4, label %bb9

bb4:                                              ; preds = %bb3
  %tmp = load i32, ptr @optind, align 4
  br i1 undef, label %bb5, label %bb7

bb5:                                              ; preds = %bb4
  %tmp6 = add nsw i32 %tmp, 1
  store i32 %tmp6, ptr @optind, align 4
  br label %bb12

bb7:                                              ; preds = %bb4
  %tmp8 = add nsw i32 %tmp, 1
  store i32 %tmp8, ptr @optind, align 4
  br label %bb13

bb9:                                              ; preds = %bb3
  %tmp10 = load i32, ptr @optind, align 4
  %tmp11 = add nsw i32 %tmp10, 1
  store i32 %tmp11, ptr @optind, align 4
  br label %bb12

bb12:                                             ; preds = %bb9, %bb5
  br label %bb13

bb13:                                             ; preds = %bb12, %bb7, %bb2
  ret void
}

@GlobalVar = internal global float 1.000000e+00

; Check that we hoist stores and remove the MSSA phi node.
; CHECK-LABEL: @hoistStoresUpdateMSSA
; CHECK: store float
; CHECK-NOT: store float
define float @hoistStoresUpdateMSSA(float %d) {
entry:
  store float 0.000000e+00, ptr @GlobalVar
  %cmp = fcmp oge float %d, 0.000000e+00
  br i1 %cmp, label %if.then, label %if.end

if.then:
  store float 0.000000e+00, ptr @GlobalVar
  br label %if.end

if.end:
  %tmp = load float, ptr @GlobalVar, align 4
  ret float %tmp
}
