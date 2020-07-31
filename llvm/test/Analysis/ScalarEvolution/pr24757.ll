; RUN: opt -analyze -enable-new-pm=0 -scalar-evolution < %s | FileCheck %s
; RUN: opt -disable-output "-passes=print<scalar-evolution>" < %s 2>&1 | FileCheck %s

; CHECK: Loop %bb1: backedge-taken count is ((2 * %a.promoted) /u 2)

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.10.0"

@a = global i8 -127, align 1
@b = common global i32 0, align 4

declare void @use(i32)

define i32 @main() {
bb:
  %a.promoted = load i8, i8* @a
  br label %bb1

bb1:                                              ; preds = %bb1, %bb
  %tmp = phi i8 [ %tmp2, %bb1 ], [ %a.promoted, %bb ]
  %tmp2 = add i8 %tmp, -1
  %tmp3 = sext i8 %tmp to i32
  %tmp4 = xor i32 %tmp3, -1
  %tmp5 = sext i8 %tmp2 to i32
  %tmpf = sub nsw i32 %tmp4, %tmp5
  %tmp6 = trunc i32 %tmpf to i8
  %tmp7 = icmp eq i8 %tmp6, 0
  br i1 %tmp7, label %bb8, label %bb1

bb8:                                              ; preds = %bb1
  store i8 %tmp2, i8* @a
  store i32 %tmp4, i32* @b
  %tmp9 = sext i8 %tmp2 to i32
  call void @use(i32 %tmp9)
  ret i32 0
}
