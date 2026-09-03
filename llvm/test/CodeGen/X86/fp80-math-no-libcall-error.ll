; RUN: not llc -mtriple=x86_64-pc-windows-msvc -filetype=null %s 2>&1 | FileCheck %s
; RUN: not llc -mtriple=x86_64-linux-android -filetype=null %s 2>&1 | FileCheck %s

; The l-suffixed long double math libcalls only exist when long double is x87.
; On windows-msvc (IEEE double) and Android (fp128) an x86_fp80 math intrinsic
; must diagnose a missing libcall rather than emit a wrongly-typed call. The
; nnan ninf variants confirm the finite-math flags do not route to a different
; (still missing) libcall.

; CHECK: error: no libcall available for facos
define x86_fp80 @test_acos(x86_fp80 %x) nounwind {
  %r = call x86_fp80 @llvm.acos.f80(x86_fp80 %x)
  ret x86_fp80 %r
}

; CHECK: error: no libcall available for fsin
define x86_fp80 @test_sin(x86_fp80 %x) nounwind {
  %r = call x86_fp80 @llvm.sin.f80(x86_fp80 %x)
  ret x86_fp80 %r
}

; CHECK: error: no libcall available for fpow
define x86_fp80 @test_pow(x86_fp80 %x, x86_fp80 %y) nounwind {
  %r = call x86_fp80 @llvm.pow.f80(x86_fp80 %x, x86_fp80 %y)
  ret x86_fp80 %r
}

; CHECK: error: no libcall available for fpow
define x86_fp80 @test_pow_finite(x86_fp80 %x, x86_fp80 %y) nounwind {
  %r = call nnan ninf x86_fp80 @llvm.pow.f80(x86_fp80 %x, x86_fp80 %y)
  ret x86_fp80 %r
}

; CHECK: error: no libcall available for fexp
define x86_fp80 @test_exp(x86_fp80 %x) nounwind {
  %r = call x86_fp80 @llvm.exp.f80(x86_fp80 %x)
  ret x86_fp80 %r
}

; CHECK: error: no libcall available for fexp
define x86_fp80 @test_exp_finite(x86_fp80 %x) nounwind {
  %r = call nnan ninf x86_fp80 @llvm.exp.f80(x86_fp80 %x)
  ret x86_fp80 %r
}

; CHECK: error: no libcall available for fexp2
define x86_fp80 @test_exp2(x86_fp80 %x) nounwind {
  %r = call x86_fp80 @llvm.exp2.f80(x86_fp80 %x)
  ret x86_fp80 %r
}

; CHECK: error: no libcall available for fexp2
define x86_fp80 @test_exp2_finite(x86_fp80 %x) nounwind {
  %r = call nnan ninf x86_fp80 @llvm.exp2.f80(x86_fp80 %x)
  ret x86_fp80 %r
}

; CHECK: error: no libcall available for flog
define x86_fp80 @test_log(x86_fp80 %x) nounwind {
  %r = call x86_fp80 @llvm.log.f80(x86_fp80 %x)
  ret x86_fp80 %r
}

; CHECK: error: no libcall available for flog
define x86_fp80 @test_log_finite(x86_fp80 %x) nounwind {
  %r = call nnan ninf x86_fp80 @llvm.log.f80(x86_fp80 %x)
  ret x86_fp80 %r
}

; CHECK: error: no libcall available for flog2
define x86_fp80 @test_log2(x86_fp80 %x) nounwind {
  %r = call x86_fp80 @llvm.log2.f80(x86_fp80 %x)
  ret x86_fp80 %r
}

; CHECK: error: no libcall available for flog2
define x86_fp80 @test_log2_finite(x86_fp80 %x) nounwind {
  %r = call nnan ninf x86_fp80 @llvm.log2.f80(x86_fp80 %x)
  ret x86_fp80 %r
}

; CHECK: error: no libcall available for flog10
define x86_fp80 @test_log10(x86_fp80 %x) nounwind {
  %r = call x86_fp80 @llvm.log10.f80(x86_fp80 %x)
  ret x86_fp80 %r
}

; CHECK: error: no libcall available for flog10
define x86_fp80 @test_log10_finite(x86_fp80 %x) nounwind {
  %r = call nnan ninf x86_fp80 @llvm.log10.f80(x86_fp80 %x)
  ret x86_fp80 %r
}

; CHECK: error: no libcall available for fldexp
define x86_fp80 @test_ldexp(x86_fp80 %x, i32 %exp) nounwind {
  %r = call x86_fp80 @llvm.ldexp.f80.i32(x86_fp80 %x, i32 %exp)
  ret x86_fp80 %r
}

; CHECK: error: no libcall available for ffrexp
define { x86_fp80, i32 } @test_frexp(x86_fp80 %x) nounwind {
  %r = call { x86_fp80, i32 } @llvm.frexp.f80.i32(x86_fp80 %x)
  ret { x86_fp80, i32 } %r
}
