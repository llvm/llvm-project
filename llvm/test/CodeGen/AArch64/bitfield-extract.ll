; RUN: llc -mtriple=aarch64-none-linux-gnu < %s | FileCheck %s

; CHECK-LABEL: @test1
; CHECK: sbfx {{x[0-9]+}}, x0, #23, #9
define i64 @test1(i32 %a) {
  %tmp = ashr i32 %a, 23
  %ext = sext i32 %tmp to i64
  %res = add i64 %ext, 1
  ret i64 %res
}

; CHECK-LABEL: @test2
; CHECK: sbfx w0, w0, #23, #8
define signext i8 @test2(i32 %a) {
  %tmp = ashr i32 %a, 23
  %res = trunc i32 %tmp to i8
  ret i8 %res
}

; CHECK-LABEL: @test3
; CHECK: sbfx w0, w0, #23, #8
define signext i8 @test3(i32 %a) {
  %tmp = lshr i32 %a, 23
  %res = trunc i32 %tmp to i8
  ret i8 %res
}

; CHECK-LABEL: @test4
; CHECK: sbfx w0, w0, #15, #16
define signext i16 @test4(i32 %a) {
  %tmp = lshr i32 %a, 15
  %res = trunc i32 %tmp to i16
  ret i16 %res
}

; CHECK-LABEL: @test5
; CHECK: sbfx w0, w0, #16, #8
define signext i8 @test5(i64 %a) {
  %tmp = lshr i64 %a, 16
  %res = trunc i64 %tmp to i8
  ret i8 %res
}

; CHECK-LABEL: @test6
; CHECK: sbfx x0, x0, #30, #8
define signext i8 @test6(i64 %a) {
  %tmp = lshr i64 %a, 30
  %res = trunc i64 %tmp to i8
  ret i8 %res
}

; CHECK-LABEL: @test7
; CHECK: sbfx x0, x0, #23, #16
define signext i16 @test7(i64 %a) {
  %tmp = lshr i64 %a, 23
  %res = trunc i64 %tmp to i16
  ret i16 %res
}

; CHECK-LABEL: @test8
; CHECK: asr w0, w0, #25
define signext i8 @test8(i32 %a) {
  %tmp = ashr i32 %a, 25
  %res = trunc i32 %tmp to i8
  ret i8 %res
}

; CHECK-LABEL: @test9
; CHECK: lsr w0, w0, #25
define signext i8 @test9(i32 %a) {
  %tmp = lshr i32 %a, 25
  %res = trunc i32 %tmp to i8
  ret i8 %res
}

; CHECK-LABEL: @test10
; CHECK: lsr x0, x0, #49
define signext i16 @test10(i64 %a) {
  %tmp = lshr i64 %a, 49
  %res = trunc i64 %tmp to i16
  ret i16 %res
}

; SHR with multiple uses is fine as SXTH and SBFX are both aliases of SBFM.
; However, allowing the transformation means the SHR and SBFX can execute in
; parallel.
;
; CHECK-LABEL: @test11
; CHECK: lsr x1, x0, #23
; CHECK: sbfx x0, x0, #23, #16
define void @test11(i64 %a) {
  %tmp = lshr i64 %a, 23
  %res = trunc i64 %tmp to i16
  call void @use(i16 %res, i64 %tmp)
  ret void
}

declare void @use(i16 signext, i64)

; CHECK-LABEL: test_complex_node:
; CHECK: ldr d0, [x0], #8
; CHECK: ubfx x[[VAL:[0-9]+]], x0, #5, #27
; CHECK: str w[[VAL]], [x2]
define <2 x i32> @test_complex_node(<2 x i32>* %addr, <2 x i32>** %addr2, i32* %bf ) {
  %vec = load <2 x i32>, <2 x i32>* %addr

  %vec.next = getelementptr <2 x i32>, <2 x i32>* %addr, i32 1
  store <2 x i32>* %vec.next, <2 x i32>** %addr2
  %lo = ptrtoint <2 x i32>* %vec.next to i32

  %val = lshr i32 %lo, 5
  store i32 %val, i32* %bf

  ret <2 x i32> %vec
}
