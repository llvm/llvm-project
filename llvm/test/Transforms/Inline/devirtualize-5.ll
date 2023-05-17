; RUN: opt -abort-on-max-devirt-iterations-reached -passes='cgscc(devirt<1>(inline,instcombine))' -S < %s | FileCheck %s
; RUN: opt -abort-on-max-devirt-iterations-reached -passes='default<O2>' -S < %s | FileCheck %s

define i32 @i() alwaysinline {
  ret i32 45
}

; CHECK-LABEL: define i32 @main
; CHECK-NEXT: ret i32 45

define i32 @main() {
  %a = alloca ptr
  store ptr @i, ptr %a
  %r = call i32 @call(ptr %a)
  ret i32 %r
}

define i32 @call(ptr %a) alwaysinline {
  %c = load ptr, ptr %a
  %r = call i32 %c()
  ret i32 %r
}
