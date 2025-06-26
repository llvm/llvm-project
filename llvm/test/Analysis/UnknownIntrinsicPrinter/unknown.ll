; RUN: opt -disable-output -passes=print-unknown-intrinsics %s 2>&1 | FileCheck %s

; ------------------------------------------------------------------------------
; Add several known intrinsics.
declare i8 @llvm.ctpop.i8(i8)
declare double @llvm.sqrt.f64(double)

; Note, the next two are known intrinsics because automatic remanging of
; intrinsics will convert them to the "correct" name.

; Remangled to llvm.sqrt.f64(double).
declare double @llvm.sqrt.foo(double)

; Remangled to llvm.ctpop.i8(i8).
declare i8 @llvm.ctpop.unknown.i8(i8)

; ------------------------------------------------------------------------------
; Add several unknown intrinsics.
declare i1 @llvm.isunordered.f32(float, float)
declare i8 @llvm.foo.unknown.i8(i8)
declare i8 @llvm.unknown.i8(i8)

; CHECK: Unknown intrinsic : llvm.isunordered.f32
; CHECK: Unknown intrinsic : llvm.foo.unknown.i8
; CHECK: Unknown intrinsic : llvm.unknown.i8
