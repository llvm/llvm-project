; RUN: opt -S -passes=declare-runtime-libcalls -mtriple=x86_64-unknown-linux-gnu < %s | FileCheck %s

define noundef nofpclass(nan) float @sqrtf(float %x) "foo" {
  %ret = call float asm "; $0 = sqrt($1)", "=r,r"(float %x)
  ret float %ret
}

; FIXME: Individual fields of nofpclass not merged
; CHECK: define noundef nofpclass(ninf nsub nnorm) float @sqrtf(float %x) [[SQRT_ATTR:#[0-9]+]] {

; CHECK: attributes [[SQRT_ATTR]] = { nocallback nofree nosync nounwind willreturn memory(errnomem: write) "foo" }
