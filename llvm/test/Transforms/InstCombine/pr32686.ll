; RUN: opt -S -instcombine %s | FileCheck %s

; CHECK-LABEL: define void @tinkywinky
; CHECK-NEXT:   %patatino = load i8, i8* @a, align 1
; CHECK-NEXT:   %tobool = icmp eq i8 %patatino, 0
; CHECK-NEXT:   %1 = zext i1 %tobool to i32
; CHECK-NEXT:   %or1 = or i32 %1, or (i32 zext (i1 icmp ne (i32* bitcast (i8* @a to i32*), i32* @b) to i32), i32 2)
; CHECK-NEXT:   store i32 %or1, i32* @b, align 4
; CHECK-NEXT:   ret void

@a = external global i8
@b = external global i32

define void @tinkywinky() {
  %patatino = load i8, i8* @a
  %tobool = icmp ne i8 %patatino, 0
  %lnot = xor i1 %tobool, true
  %lnot.ext = zext i1 %lnot to i32
  %or = or i32 xor (i32 zext (i1 icmp ne (i32* bitcast (i8* @a to i32*), i32* @b) to i32), i32 2), %lnot.ext
  store i32 %or, i32* @b, align 4
  ret void
}
