; RUN: not llc -mtriple=nvptx64 -filetype=null %s 2>&1 | FileCheck %s

; NVPTX has no fp128 soft-float library, so these operations have no libcall
; available. Make sure they emit a proper diagnostic rather than fatal erroring
; in makeLibCall.

; CHECK: error: no libcall available for fexp10 with type f128
define fp128 @test_exp10(fp128 %x) nounwind {
  %r = call fp128 @llvm.exp10.f128(fp128 %x)
  ret fp128 %r
}

; CHECK: error: no libcall available for fmaximum with type f128
define fp128 @test_maximum(fp128 %x, fp128 %y) nounwind {
  %r = call fp128 @llvm.maximum.f128(fp128 %x, fp128 %y)
  ret fp128 %r
}

; CHECK: error: no libcall available for fma with type f128
define fp128 @test_fma(fp128 %x, fp128 %y, fp128 %z) nounwind {
  %r = call fp128 @llvm.fma.f128(fp128 %x, fp128 %y, fp128 %z)
  ret fp128 %r
}

; CHECK: error: no libcall available for strict_fma with type f128
define fp128 @test_strict_fma(fp128 %x, fp128 %y, fp128 %z) nounwind strictfp {
  %r = call fp128 @llvm.experimental.constrained.fma.f128(fp128 %x, fp128 %y, fp128 %z, metadata !"round.tonearest", metadata !"fpexcept.strict")
  ret fp128 %r
}

; CHECK: error: no libcall available for ffrexp with type f128
define { fp128, i32 } @test_frexp(fp128 %x) nounwind {
  %r = call { fp128, i32 } @llvm.frexp.f128.i32(fp128 %x)
  ret { fp128, i32 } %r
}

; CHECK: error: no libcall available for lrint with type f128
define i32 @test_lrint(fp128 %x) nounwind {
  %r = call i32 @llvm.lrint.i32.f128(fp128 %x)
  ret i32 %r
}

; CHECK: error: no libcall available for strict_fadd with type f128
define fp128 @test_strict_fadd(fp128 %x, fp128 %y) nounwind strictfp {
  %r = call fp128 @llvm.experimental.constrained.fadd.f128(fp128 %x, fp128 %y, metadata !"round.tonearest", metadata !"fpexcept.strict")
  ret fp128 %r
}
