; RUN: opt -passes=sccp -S < %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-apple-darwin10.0"

define void @test1(i8 %arg) {
entry:
  br i1 undef, label %return, label %bb

bb:   
  br label %bb34

bb23: 
  %c = icmp eq i8 %arg, undef 
  br i1 %c, label %bb34, label %bb23

bb34:
  %Kind.1 = phi i32 [ undef, %bb ], [ %ins174, %bb23 ] 
  %mask173 = or i32 %Kind.1, 7
  %ins174 = and i32 %mask173, -249
  br label %bb23

return:
  ret void
}

define i32 @test2([4 x i32] %A) {
  %B = extractvalue [4 x i32] %A, 1
  ret i32 %B
}

define x86_mmx @test3() {
  %load = load x86_mmx, ptr null
  ret x86_mmx %load
}
