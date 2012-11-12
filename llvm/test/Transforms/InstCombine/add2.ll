; RUN: opt < %s -instcombine -S | FileCheck %s

define i64 @test1(i64 %A, i32 %B) {
        %tmp12 = zext i32 %B to i64
        %tmp3 = shl i64 %tmp12, 32
        %tmp5 = add i64 %tmp3, %A
        %tmp6 = and i64 %tmp5, 123
        ret i64 %tmp6
; CHECK: @test1
; CHECK-NEXT: and i64 %A, 123
; CHECK-NEXT: ret i64
}

define i32 @test2(i32 %A) {
  %B = and i32 %A, 7
  %C = and i32 %A, 32
  %F = add i32 %B, %C
  ret i32 %F
; CHECK: @test2
; CHECK-NEXT: and i32 %A, 39
; CHECK-NEXT: ret i32
}

define i32 @test3(i32 %A) {
  %B = and i32 %A, 128
  %C = lshr i32 %A, 30
  %F = add i32 %B, %C
  ret i32 %F
; CHECK: @test3
; CHECK-NEXT: and
; CHECK-NEXT: lshr
; CHECK-NEXT: or i32 %B, %C
; CHECK-NEXT: ret i32
}

define i32 @test4(i32 %A) {
  %B = add nuw i32 %A, %A
  ret i32 %B
; CHECK: @test4
; CHECK-NEXT: %B = shl nuw i32 %A, 1
; CHECK-NEXT: ret i32 %B
}

