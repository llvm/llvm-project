; RUN: opt < %s -passes=instcombine -data-layout="p:32:32" -S | FileCheck %s --check-prefixes=CHECK,CHECK32
; RUN: opt < %s -passes=instcombine -data-layout="p:64:64" -S | FileCheck %s --check-prefixes=CHECK,CHECK64

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
  call void bitcast (void (i32, ...)* @d to void (i32, i32*)*)(i32 0, i32* nocapture %y)
  call void bitcast (void (i32, ...)* @d to void (i32*)*)(i32* nocapture noundef %y)
  ret void
}
; CHECK-LABEL: define void @g(i32* %y)
; CHECK:    call i32 bitcast (i32 (i32*)* @b to i32 (i32)*)(i32 zeroext 0)
; CHECK:    call void (...) @c(i32* %y)
; CHECK:    call void bitcast (void (...)* @c to void (i32*)*)(i32* sret(i32) %y)
; CHECK:    call void bitcast (void (i32, ...)* @d to void (i32, i32*)*)(i32 0, i32* sret(i32) %y)
; CHECK:    call void (i32, ...) @d(i32 0, i32* nocapture %y)
; CHECK32:  %2 = ptrtoint i32* %y to i32
; CHECK32:  call void (i32, ...) @d(i32 noundef %2)
; CHECK64:  call void bitcast (void (i32, ...)* @d to void (i32*)*)(i32* nocapture noundef %y)
