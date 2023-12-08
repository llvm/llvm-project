; RUN: opt < %s -data-layout="E-p:32:32" -passes=ipsccp -S | FileCheck %s

@j = internal global i32 undef, align 4

; Make sure we do not mark loads from undef as overdefined.
define i32 @test5(i32 %b) {
; CHECK-LABEL: define i32 @test5(i32 %b)
; CHECK-NEXT:    %add = add nsw i32 undef, %b
; CHECK-NEXT:    ret i32 %add
;
  %l = load i32, ptr @j, align 4
  %add = add nsw i32 %l, %b
  ret i32 %add
}
