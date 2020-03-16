; RUN: opt < %s -instcombine -S | FileCheck %s

target datalayout = "e-p:32:32"
target triple = "i686-pc-win32"


declare token @llvm.call.preallocated.setup(i32)
declare i8* @llvm.call.preallocated.arg(token, i32)

declare void @takes_i32(i32)
declare void @takes_i32_preallocated(i32* preallocated(i32))

define void @f() {
; CHECK-LABEL: define void @f()
  %t = call token @llvm.call.preallocated.setup(i32 1)
  %a = call i8* @llvm.call.preallocated.arg(token %t, i32 0) preallocated(i32)
  %arg = bitcast i8* %a to i32*
  call void bitcast (void (i32)* @takes_i32 to void (i32*)*)(i32* preallocated(i32) %arg) ["preallocated"(token %t)]
; CHECK: call void bitcast{{.*}}@takes_i32
  ret void
}

define void @g() {
; CHECK-LABEL: define void @g()
  call void bitcast (void (i32*)* @takes_i32_preallocated to void (i32)*)(i32 0)
; CHECK: call void bitcast{{.*}}@takes_i32_preallocated
  ret void
}
