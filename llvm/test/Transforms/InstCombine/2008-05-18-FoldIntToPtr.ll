; RUN: opt < %s -instcombine -S | grep "ret i1 false" | count 2
; PR2329

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32"
target triple = "i386-pc-linux-gnu"

define i1 @f1() {
  ret i1 icmp eq (i8* inttoptr (i32 1 to i8*), i8* inttoptr (i32 2 to i8*))
}

define i1 @f2() {
  ret i1 icmp eq (i8* inttoptr (i16 1 to i8*), i8* inttoptr (i16 2 to i8*))
}
