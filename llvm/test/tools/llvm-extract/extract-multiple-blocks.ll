; RUN: llvm-extract -S -bb foo:bb4 -bb foo:bb7 %s | FileCheck %s

; CHECK: @foo.bb4
; CHECK: @foo.bb7
define i32 @foo(i32 %arg) {
bb:
  %tmp = alloca i32, align 4
  %tmp1 = alloca i32, align 4
  store i32 %arg, i32* %tmp1, align 4
  %tmp2 = load i32, i32* %tmp1, align 4
  %tmp3 = icmp sgt i32 %tmp2, 0
  br i1 %tmp3, label %bb4, label %bb7

bb4:                                              ; preds = %bb
  %tmp5 = load i32, i32* %tmp1, align 4
  %tmp6 = add nsw i32 %tmp5, 1
  store i32 %tmp6, i32* %tmp1, align 4
  store i32 %tmp6, i32* %tmp, align 4
  br label %bb8

bb7:                                              ; preds = %bb
  store i32 0, i32* %tmp, align 4
  br label %bb8

bb8:                                              ; preds = %bb7, %bb4
  %tmp9 = load i32, i32* %tmp, align 4
  ret i32 %tmp9
}

