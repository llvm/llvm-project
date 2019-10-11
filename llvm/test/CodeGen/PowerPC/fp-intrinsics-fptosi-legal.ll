; RUN: llc -O3 -mtriple=powerpc-unknown-linux-gnu -mcpu=e500 -mattr=spe < %s | FileCheck %s

; PowerPC SPE is a rare in-tree target that has the FP_TO_SINT node marked
; as Legal.

; Verify that fptosi(42.1) isn't simplified when the rounding mode is
; unknown.
; Verify that no gross errors happen.
; CHECK-LABEL: @f20
; COMMON: cfdctsiz
define i32 @f20(double %a) strictfp {
entry:
  %result = call i32 @llvm.experimental.constrained.fptosi.i32.f64(double 42.1,
                                               metadata !"fpexcept.strict")
                                               strictfp
  ret i32 %result
}

@llvm.fp.env = thread_local global i8 zeroinitializer, section "llvm.metadata"
declare i32 @llvm.experimental.constrained.fptosi.i32.f64(double, metadata)
