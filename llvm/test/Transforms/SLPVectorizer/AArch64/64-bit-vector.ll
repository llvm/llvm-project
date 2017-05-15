; RUN: opt -S -slp-vectorizer -mtriple=aarch64--linux-gnu -mcpu=generic < %s | FileCheck %s
; RUN: opt -S -slp-vectorizer -mtriple=aarch64-apple-ios -mcpu=cyclone < %s | FileCheck %s
; Currently disabled for a few subtargets (e.g. Kryo):
; RUN: opt -S -slp-vectorizer -mtriple=aarch64--linux-gnu -mcpu=kryo < %s | FileCheck --check-prefix=NO_SLP %s
; RUN: opt -S -slp-vectorizer -mtriple=aarch64--linux-gnu -mcpu=generic -slp-min-reg-size=128 < %s | FileCheck --check-prefix=NO_SLP %s

define void @f(float* %r, float* %w) {
  %r0 = getelementptr inbounds float, float* %r, i64 0
  %r1 = getelementptr inbounds float, float* %r, i64 1
  %f0 = load float, float* %r0
  %f1 = load float, float* %r1
  %add0 = fadd float %f0, %f0
; CHECK:  fadd <2 x float>
; NO_SLP: fadd float
; NO_SLP: fadd float
  %add1 = fadd float %f1, %f1
  %w0 = getelementptr inbounds float, float* %w, i64 0
  %w1 = getelementptr inbounds float, float* %w, i64 1
  store float %add0, float* %w0
  store float %add1, float* %w1
  ret void
}
