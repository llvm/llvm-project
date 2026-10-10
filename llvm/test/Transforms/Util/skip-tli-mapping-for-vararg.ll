; NOTE: inject-tli-mappings must not process variadic libcalls. Without an early
; return for isVarArg(), the pass can reach addVariantDeclaration(), which
; does not support vararg callees.
; For @sin, the declaration must be double (double, ...) to match
; call double (double, ...) @sin; declare double (...) alone is double (...),
; so getCalledFunction() is null and the isVarArg guard is not reached.
; For a K&R int pow() / return pow() pattern, clang emits
; declare i32 @pow(...) and call i32 (...) @pow(), which match and are vararg.

; RUN: split-file %s %t
; RUN: opt -mtriple=x86_64-unknown-linux-gnu -vector-library=AMDLIBM -passes=inject-tli-mappings -S < %t/scalar-sin.ll | FileCheck %s --check-prefix=SCALAR
; RUN: opt -mtriple=x86_64-unknown-linux-gnu -vector-library=AMDLIBM -passes=inject-tli-mappings -S < %t/vararg-sin.ll -o %t/b.ll
; RUN: FileCheck %s --check-prefix=VARARG --implicit-check-not=vector-function-abi-variant --implicit-check-not=@llvm.compiler.used --implicit-check-not=@amd_vrd2_sin < %t/b.ll
; RUN: opt -mtriple=x86_64-unknown-linux-gnu -vector-library=AMDLIBM -passes=inject-tli-mappings -S < %t/vararg-pow-noargs.ll -o %t/c.ll
; RUN: FileCheck %s --check-prefix=POW --implicit-check-not=vector-function-abi-variant --implicit-check-not=@llvm.compiler.used < %t/c.ll

; SCALAR: @llvm.compiler.used
; SCALAR: @amd_vrd2_sin
; SCALAR: define double @scalar_sin
; SCALAR: call double @sin(double
; SCALAR: "vector-function-abi-variant"=
; VARARG-LABEL: @vararg_sin_callee(
; VARARG: call double (double, ...) @sin(
; POW: declare i32 @pow
; POW: call i32 (...) @pow(

;--- scalar-sin.ll
declare double @sin(double)

define double @scalar_sin(double %x) {
  %r = call double @sin(double %x)
  ret double %r
}

;--- vararg-sin.ll
declare double @sin(double, ...)

define i32 @vararg_sin_callee(double %x) {
  %r = call double (double, ...) @sin(double %x)
  %i = fptosi double %r to i32
  ret i32 %i
}

;--- vararg-pow-noargs.ll
declare i32 @pow(...)

define i32 @call_pow_noargs() {
  %r = call i32 (...) @pow()
  ret i32 %r
}
