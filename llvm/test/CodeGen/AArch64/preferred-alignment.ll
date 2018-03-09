; RUN: llc -mtriple=aarch64 -O0 -fast-isel < %s | FileCheck %s

; Function Attrs: nounwind
define i32 @foo() #0 {
entry:
  %c = alloca i8, align 1
; CHECK:	add	x0, sp, #12
  %s = alloca i16, align 2
; CHECK-NEXT:	add	x1, sp, #8
  %i = alloca i32, align 4
; CHECK-NEXT:	add	x2, sp, #4
  %call = call i32 @bar(i8* %c, i16* %s, i32* %i)
  %0 = load i8, i8* %c, align 1
  %conv = zext i8 %0 to i32
  %add = add nsw i32 %call, %conv
  %1 = load i16, i16* %s, align 2
  %conv1 = sext i16 %1 to i32
  %add2 = add nsw i32 %add, %conv1
  %2 = load i32, i32* %i, align 4
  %add3 = add nsw i32 %add2, %2
  ret i32 %add3
}

declare i32 @bar(i8*, i16*, i32*) #1

attributes #0 = { nounwind "no-frame-pointer-elim"="false" }
attributes #1 = { "no-frame-pointer-elim"="false" }

