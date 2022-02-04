; RUN: opt < %s -passes=instcombine -S | FileCheck %s

define signext i32 @b(i32* inreg %x)   {
  ret i32 0
}

define void @c(...) {
  ret void
}

declare void @useit(i32)

define void @d(i32 %x, ...) {
  call void @useit(i32 %x)
  ret void
}

define void @g(i32* %y) {
  call i32 bitcast (i32 (i32*)* @b to i32 (i32)*)(i32 zeroext 0)
  call void bitcast (void (...)* @c to void (i32*)*)(i32* %y)
  call void bitcast (void (...)* @c to void (i32*)*)(i32* sret(i32) %y)
  call void bitcast (void (i32, ...)* @d to void (i32, i32*)*)(i32 0, i32* sret(i32) %y)
  ret void
}
; CHECK-LABEL: define void @g(i32* %y)
; CHECK: call i32 bitcast (i32 (i32*)* @b to i32 (i32)*)(i32 zeroext 0)
; CHECK: call void (...) @c(i32* %y)
; CHECK: call void bitcast (void (...)* @c to void (i32*)*)(i32* sret(i32) %y)
; CHECK: call void bitcast (void (i32, ...)* @d to void (i32, i32*)*)(i32 0, i32* sret(i32) %y)
