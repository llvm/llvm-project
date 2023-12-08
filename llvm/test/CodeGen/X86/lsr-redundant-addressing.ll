; RUN: llc -mtriple=x86_64-- < %s | FileCheck %s
; rdar://9081094

; LSR shouldn't create lots of redundant address computations.

%0 = type { i32, [3 x i32] }
%1 = type { ptr, i32, i32, [3 x i32], ptr, ptr, ptr }

@pgm = external hidden unnamed_addr global [5 x %0], align 32
@isa = external hidden unnamed_addr constant [13 x %1], align 32

define void @main_bb.i() nounwind {
; CHECK: main_bb.i:
; CHECK-NOT: ret
; CHECK: addq $-16,
; CHECK-NOT: ret
; CHECK: ret

bb:
  br label %bb38

bb38:                                             ; preds = %bb200, %bb
  %tmp39 = phi i64 [ %tmp201, %bb200 ], [ 0, %bb ]
  %tmp40 = sub i64 0, %tmp39
  %tmp47 = getelementptr [5 x %0], ptr @pgm, i64 0, i64 %tmp40, i32 0
  %tmp34 = load i32, ptr %tmp47, align 16
  %tmp203 = icmp slt i32 %tmp34, 12
  br i1 %tmp203, label %bb215, label %bb200

bb200:                                            ; preds = %bb38
  %tmp201 = add i64 %tmp39, 1
  br label %bb38

bb215:                                            ; preds = %bb38
  %tmp50 = getelementptr [5 x %0], ptr @pgm, i64 0, i64 %tmp40, i32 1, i64 2
  %tmp49 = getelementptr [5 x %0], ptr @pgm, i64 0, i64 %tmp40, i32 1, i64 1
  %tmp48 = getelementptr [5 x %0], ptr @pgm, i64 0, i64 %tmp40, i32 1, i64 0
  %tmp216 = add nsw i32 %tmp34, 1
  store i32 %tmp216, ptr %tmp47, align 16
  %tmp217 = sext i32 %tmp216 to i64
  %tmp218 = getelementptr inbounds [13 x %1], ptr @isa, i64 0, i64 %tmp217, i32 3, i64 0
  %tmp219 = load i32, ptr %tmp218, align 8
  store i32 %tmp219, ptr %tmp48, align 4
  %tmp220 = getelementptr inbounds [13 x %1], ptr @isa, i64 0, i64 %tmp217, i32 3, i64 1
  %tmp221 = load i32, ptr %tmp220, align 4
  store i32 %tmp221, ptr %tmp49, align 4
  %tmp222 = getelementptr inbounds [13 x %1], ptr @isa, i64 0, i64 %tmp217, i32 3, i64 2
  %tmp223 = load i32, ptr %tmp222, align 8
  store i32 %tmp223, ptr %tmp50, align 4
  ret void
}
