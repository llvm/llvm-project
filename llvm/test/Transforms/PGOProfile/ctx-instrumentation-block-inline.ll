; RUN: opt -passes=noinline-nonprevailing -S < %s 2>&1 | FileCheck %s

define void @a() {
  ret void
}

define void @b() #0 {
  ret void
}

define weak_odr void @c() {
  ret void
}

define weak_odr void @d() #0{
  ret void
}

attributes #0 = { alwaysinline }

; CHECK: void @a() {
; CHECK: void @b() #0
; CHECK: void @c() #1
; CHECK: void @d() #1
; CHECK: attributes #1 = { noinline }
