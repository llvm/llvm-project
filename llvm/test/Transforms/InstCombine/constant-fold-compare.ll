; RUN: opt < %s -passes=instcombine -S | FileCheck %s
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32-n8:16:32"

define i32 @a() nounwind readnone {
entry:
  ret i32 zext (i1 icmp eq (i32 0, i32 ptrtoint (i32 ()* @a to i32)) to i32)
}
; CHECK: ret i32 0
