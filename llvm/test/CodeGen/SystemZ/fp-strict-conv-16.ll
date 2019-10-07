; Test f128 floating-point strict conversion to/from integers on z14.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z14 | FileCheck %s

; FIXME: llvm.experimental.constrained.[su]itofp does not yet exist

declare i32 @llvm.experimental.constrained.fptosi.i32.f128(fp128, metadata)
declare i64 @llvm.experimental.constrained.fptosi.i64.f128(fp128, metadata)

declare i32 @llvm.experimental.constrained.fptoui.i32.f128(fp128, metadata)
declare i64 @llvm.experimental.constrained.fptoui.i64.f128(fp128, metadata)

; Test signed f128->i32.
define i32 @f5(fp128 *%src) #0 {
; CHECK-LABEL: f5:
; CHECK: vl %v0, 0(%r2)
; CHECK: vrepg %v2, %v0, 1
; CHECK: cfxbr %r2, 5, %f0
; CHECK: br %r14
  %f = load fp128, fp128 *%src
  %conv = call i32 @llvm.experimental.constrained.fptosi.i32.f128(fp128 %f,
                                               metadata !"fpexcept.strict") #0
  ret i32 %conv
}

; Test signed f128->i64.
define i64 @f6(fp128 *%src) #0 {
; CHECK-LABEL: f6:
; CHECK: vl %v0, 0(%r2)
; CHECK: vrepg %v2, %v0, 1
; CHECK: cgxbr %r2, 5, %f0
; CHECK: br %r14
  %f = load fp128, fp128 *%src
  %conv = call i64 @llvm.experimental.constrained.fptosi.i64.f128(fp128 %f,
                                               metadata !"fpexcept.strict") #0
  ret i64 %conv
}

; Test unsigned f128->i32.
define i32 @f7(fp128 *%src) #0 {
; CHECK-LABEL: f7:
; CHECK: vl %v0, 0(%r2)
; CHECK: vrepg %v2, %v0, 1
; CHECK: clfxbr %r2, 5, %f0, 0
; CHECK: br %r14
  %f = load fp128, fp128 *%src
  %conv = call i32 @llvm.experimental.constrained.fptoui.i32.f128(fp128 %f,
                                               metadata !"fpexcept.strict") #0
  ret i32 %conv
}

; Test unsigned f128->i64.
define i64 @f8(fp128 *%src) #0 {
; CHECK-LABEL: f8:
; CHECK: vl %v0, 0(%r2)
; CHECK: vrepg %v2, %v0, 1
; CHECK: clgxbr %r2, 5, %f0, 0
; CHECK: br %r14
  %f = load fp128, fp128 *%src
  %conv = call i64 @llvm.experimental.constrained.fptoui.i64.f128(fp128 %f,
                                               metadata !"fpexcept.strict") #0
  ret i64 %conv
}

attributes #0 = { strictfp }
