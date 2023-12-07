; RUN: llc -mtriple=aarch64-linux-gnu -O3 -o - %s | FileCheck %s

@a = external global i8, align 1
@b = external global i8, align 1

; Test that SUBS is replaced by SUB if condition flags are not used.
define i32 @test01() nounwind {
; CHECK: ldrb {{.*}}
; CHECK-NEXT: ldrb {{.*}}
; CHECK: sub {{.*}}
; CHECK-NEXT: cmn {{.*}}
entry:
  %0 = load i8, ptr @a, align 1
  %conv = zext i8 %0 to i32
  %1 = load i8, ptr @b, align 1
  %conv1 = zext i8 %1 to i32
  %s = sub nsw i32 %conv1, %conv
  %cmp0 = icmp eq i32 %s, -1
  %cmp1 = sext i1 %cmp0 to i8
  store i8 %cmp1, ptr @a
  ret i32 0
}

