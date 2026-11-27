; REQUIRES: system-linux
; REQUIRES: systemz-registered-target
; RUN: %lli -jit-kind=orc-lazy %s | FileCheck %s

; Verifies that a vector argument survives the lazy-call resolver. The callee
; is compiled on its first call, and compilation clobbers v24-v31, so the
; resolver must save and restore them for the sum to be correct.

@vec = global <4 x i32> <i32 1, i32 2, i32 3, i32 4>
@fmt = private constant [4 x i8] c"%d\0A\00"

declare i32 @printf(ptr, ...)

; Sums the four lanes. The vector arrives in %v24 under the s390x ABI.
define i32 @vsum(<4 x i32> %v) noinline {
  %e0 = extractelement <4 x i32> %v, i32 0
  %e1 = extractelement <4 x i32> %v, i32 1
  %e2 = extractelement <4 x i32> %v, i32 2
  %e3 = extractelement <4 x i32> %v, i32 3
  %s01 = add i32 %e0, %e1
  %s23 = add i32 %e2, %e3
  %s = add i32 %s01, %s23
  ret i32 %s
}

define i32 @main() {
  ; volatile stops the load from being folded to a constant.
  %v = load volatile <4 x i32>, ptr @vec
  %r = call i32 @vsum(<4 x i32> %v)
  call i32 (ptr, ...) @printf(ptr @fmt, i32 %r)
  ret i32 0
}

; CHECK: 10
