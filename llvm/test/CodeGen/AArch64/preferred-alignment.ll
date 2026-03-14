; RUN: llc -mtriple=aarch64 -O0 -fast-isel < %s | FileCheck %s

; Function Attrs: nounwind
define i32 @foo() #0 {
entry:
  %c = alloca i8
; CHECK:	add	x0, sp, #12
  %s = alloca i16
; CHECK-NEXT:	add	x1, sp, #8
  %i = alloca i32
; CHECK-NEXT:	add	x2, sp, #4
  %call = call i32 @bar(ptr %c, ptr %s, ptr %i)
  %0 = load i8, ptr %c, align 1
  %conv = zext i8 %0 to i32
  %add = add nsw i32 %call, %conv
  %1 = load i16, ptr %s, align 2
  %conv1 = sext i16 %1 to i32
  %add2 = add nsw i32 %add, %conv1
  %2 = load i32, ptr %i, align 4
  %add3 = add nsw i32 %add2, %2
  ret i32 %add3
}

declare i32 @bar(ptr, ptr, ptr) #1

attributes #0 = { nounwind "frame-pointer"="none" }
attributes #1 = { "frame-pointer"="none" }

