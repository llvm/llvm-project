; RUN: opt -opaque-pointers=0 < %s -passes=instcombine -S | FileCheck %s

target datalayout = "e-p:32:32"
target triple = "i686-pc-linux-gnu"

declare void @takes_i32(i32)
declare void @takes_i32_inalloca(i32* inalloca(i32))

define void @f() {
; CHECK-LABEL: define void @f()
  %args = alloca inalloca i32
  call void bitcast (void (i32)* @takes_i32 to void (i32*)*)(i32* inalloca(i32) %args)
; CHECK: call void bitcast
  ret void
}

define void @g() {
; CHECK-LABEL: define void @g()
  call void bitcast (void (i32*)* @takes_i32_inalloca to void (i32)*)(i32 0)
; CHECK: call void bitcast
  ret void
}
