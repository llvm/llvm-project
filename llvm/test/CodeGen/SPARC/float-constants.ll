; RUN: llc < %s -march=sparc | FileCheck %s
; RUN: llc < %s -march=sparcel | FileCheck %s --check-prefix=CHECK-LE

;; Bitcast should not do a runtime conversion, but rather emit a
;; constant into integer registers directly.

; CHECK-LABEL: bitcast:
; CHECK: sethi 1049856, %o0
; CHECK: mov %g0, %o1
; CHECK-LE: mov %g0, %o0
; CHECK-LE: sethi 1049856, %o1
define <2 x i32> @bitcast() {
  %1 = bitcast double 5.0 to <2 x i32>
  ret <2 x i32> %1
}

;; Same thing for a call using a double (which gets passed in integer
;; registers)

; CHECK-LABEL: test_call
; CHECK: sethi 1049856, %o0
; CHECK: mov %g0, %o1
; CHECK-LE: mov %g0, %o0
; CHECK-LE: sethi 1049856, %o1
declare void @a(double)
define void @test_call() {
  call void @a(double 5.0)
  ret void
}

;; And for a libcall emitted from the pow intrinsic.  (libcall
;; emission happens after SelectionDAG type legalization, so is a bit
;; different than a normal function call. This was crashing before,
;; due to an earlier broken workaround for this issue.)

; CHECK-LABEL: test_intrins_call
; CHECK: sethi 1048576, %o0
; CHECK: mov %g0, %o1
; CHECK: mov %o0, %o2
; CHECK: mov %o1, %o3
; CHECK-LE: mov %g0, %o0
; CHECK-LE: sethi 1048576, %o1
declare double @llvm.pow.f64(double, double)
define double @test_intrins_call() {
  %1 = call double @llvm.pow.f64(double 2.0, double 2.0)
  ret double %1
}
