; Test f128 floating-point strict conversion to/from integers on z14.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z14 | FileCheck %s

declare fp128 @llvm.experimental.constrained.sitofp.f128.i32(i32, metadata, metadata)
declare fp128 @llvm.experimental.constrained.sitofp.f128.i64(i64, metadata, metadata)

declare fp128 @llvm.experimental.constrained.uitofp.f128.i32(i32, metadata, metadata)
declare fp128 @llvm.experimental.constrained.uitofp.f128.i64(i64, metadata, metadata)

declare i32 @llvm.experimental.constrained.fptosi.i32.f128(fp128, metadata)
declare i64 @llvm.experimental.constrained.fptosi.i64.f128(fp128, metadata)

declare i32 @llvm.experimental.constrained.fptoui.i32.f128(fp128, metadata)
declare i64 @llvm.experimental.constrained.fptoui.i64.f128(fp128, metadata)

; Test signed i32->f128.
define void @f1(i32 %i, fp128 *%dst) #0 {
; CHECK-LABEL: f1:
; CHECK: cxfbr %f0, %r2
; CHECK: vmrhg %v0, %v0, %v2
; CHECK: vst %v0, 0(%r3)
; CHECK: br %r14
  %conv = call fp128 @llvm.experimental.constrained.sitofp.f128.i32(i32 %i,
                                               metadata !"round.dynamic",
                                               metadata !"fpexcept.strict") #0
  store fp128 %conv, fp128 *%dst
  ret void
}

; Test signed i64->f128.
define void @f2(i64 %i, fp128 *%dst) #0 {
; CHECK-LABEL: f2:
; CHECK: cxgbr %f0, %r2
; CHECK: vmrhg %v0, %v0, %v2
; CHECK: vst %v0, 0(%r3)
; CHECK: br %r14
  %conv = call fp128 @llvm.experimental.constrained.sitofp.f128.i64(i64 %i,
                                               metadata !"round.dynamic",
                                               metadata !"fpexcept.strict") #0
  store fp128 %conv, fp128 *%dst
  ret void
}

; Test unsigned i32->f128.
define void @f3(i32 %i, fp128 *%dst) #0 {
; CHECK-LABEL: f3:
; CHECK: cxlfbr %f0, 0, %r2, 0
; CHECK: vmrhg %v0, %v0, %v2
; CHECK: vst %v0, 0(%r3)
; CHECK: br %r14
  %conv = call fp128 @llvm.experimental.constrained.uitofp.f128.i32(i32 %i,
                                               metadata !"round.dynamic",
                                               metadata !"fpexcept.strict") #0
  store fp128 %conv, fp128 *%dst
  ret void
}

; Test unsigned i64->f128.
define void @f4(i64 %i, fp128 *%dst) #0 {
; CHECK-LABEL: f4:
; CHECK: cxlgbr %f0, 0, %r2, 0
; CHECK: vmrhg %v0, %v0, %v2
; CHECK: vst %v0, 0(%r3)
; CHECK: br %r14
  %conv = call fp128 @llvm.experimental.constrained.uitofp.f128.i64(i64 %i,
                                               metadata !"round.dynamic",
                                               metadata !"fpexcept.strict") #0
  store fp128 %conv, fp128 *%dst
  ret void
}

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
