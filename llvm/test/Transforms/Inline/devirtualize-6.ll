; Make sure we don't detect devirtualization on inlining a function with a direct call
; RUN: opt -abort-on-max-devirt-iterations-reached -passes='cgscc(devirt<0>(inline))' -S < %s | FileCheck %s

define i32 @i() noinline {
  ret i32 45
}

; CHECK-NOT: call i32 @call()

define i32 @main() {
  %r = call i32 @call()
  ret i32 %r
}

define i32 @call() alwaysinline {
  %r = call i32 @i()
  ret i32 %r
}
