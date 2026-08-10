; RUN: opt < %s -passes=inline | llvm-dis

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin10.0"

define i32 @main(i32 %argc, ptr %argv) nounwind ssp {
entry:
  call fastcc void @c() nounwind
  unreachable
}

define internal fastcc void @a() nounwind ssp {
entry:
  %al = alloca [3 x i32], align 4
  %0 = getelementptr inbounds [3 x i32], ptr %al, i32 0, i32 2

  call fastcc void @c() nounwind
  unreachable
}

define internal fastcc void @b() nounwind ssp {
entry:
  tail call fastcc void @a() nounwind ssp
  unreachable
}

define internal fastcc void @c() nounwind ssp {
entry:
  call fastcc void @b() nounwind
  unreachable
}
