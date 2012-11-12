; RUN: llc %s -o - | grep "__TEXT,__const"
; PR5929
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128-n8:16:32"
target triple = "i386-apple-darwin10.0"

; This array should go into the __TEXT,__const section, not into the
; __DATA,__const section, because the elements don't need relocations.
@test.array = internal unnamed_addr constant [3 x i32] [i32 sub (i32 ptrtoint (i8* blockaddress(@test, %foo) to i32), i32 ptrtoint (i8* blockaddress(@test, %foo) to i32)), i32 sub (i32 ptrtoint (i8* blockaddress(@test, %bar) to i32), i32 ptrtoint (i8* blockaddress(@test, %foo) to i32)), i32 sub (i32 ptrtoint (i8* blockaddress(@test, %hack) to i32), i32 ptrtoint (i8* blockaddress(@test, %foo) to i32))] ; <[3 x i32]*> [#uses=1]

define void @test(i32 %i) nounwind ssp {
entry:
  call void @test(i32 1)
  br label %foo

foo:
  call void @test(i32 1)
  br label %bar

bar:
  call void @test(i32 1)
  br label %hack

hack:
  call void @test(i32 1)
  ret void
}
