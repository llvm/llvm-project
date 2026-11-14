; REQUIRES: x86-registered-target
; RUN: opt -S -passes=declare-runtime-libcalls -mtriple=x86_64-unknown-linux-gnu < %s | FileCheck %s

; Check an already declared function
; CHECK: declare float @logf(float)
declare float @logf(float)

; Check an already defined function
; CHECK: define float @sinf(float %x) {
define float @sinf(float %x) {
  ret float %x
}

; CHECK: declare void @_Unwind_Resume(ptr)

; CHECK: declare void @__atomic_load(i64, ptr captures(none), ptr writeonly captures(none), i32) [[ATOMIC_ATTRS:#[0-9]+]]
; CHECK: declare i32 @__atomic_load_4(ptr captures(none), i32) [[ATOMIC_ATTRS]]

; CHECK: declare void @__llvm_memcpy_element_unordered_atomic_1(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64) [[ARGMEM_ATTRS:#[0-9]+]]

; CHECK: declare ptr @__memcpy_chk(ptr, ptr, i64, i64)
; CHECK: declare ptr @__memmove_chk(ptr, ptr, i64, i64)
; CHECK: declare ptr @__memset_chk(ptr, i32, i64, i64)

; CHECK: declare i32 @__mulosi4(i32, i32, ptr noalias nonnull writeonly align 4 dereferenceable(4)) [[ARGMEM_WRITE_ATTRS:#[0-9]+]]

; CHECK: declare i128 @__umodti3(i128, i128) [[NOMEM_ATTRS:#[0-9]+]]

; CHECK: declare float @acosf(float) [[ERRNO_ATTRS:#[0-9]+]]

; CHECK: declare noalias noundef ptr @calloc(i64 noundef, i64 noundef) [[CALLOC_ATTRS:#[0-9]+]]

; CHECK: declare double @fdim(double, double) [[ERRNO_ATTRS]]
; CHECK: declare float @fdimf(float, float) [[ERRNO_ATTRS]]
; CHECK: declare x86_fp80 @fdiml(x86_fp80, x86_fp80) [[ERRNO_ATTRS]]

; CHECK: declare void @free(ptr allocptr noundef captures(none)) [[FREE_ATTRS:#[0-9]+]]

; CHECK: declare noalias noundef ptr @malloc(i64 noundef) [[MALLOC_ATTRS:#[0-9]+]]


; CHECK: declare nofpclass(snan inf zero sub norm) double @nan(ptr readonly captures(none)) [[ARGMEM_READ_ATTRS:#[0-9]+]]
; CHECK: declare nofpclass(snan inf zero sub norm) float @nanf(ptr readonly captures(none)) [[ARGMEM_READ_ATTRS]]
; CHECK: declare nofpclass(snan inf zero sub norm) x86_fp80 @nanl(ptr readonly captures(none)) [[ARGMEM_READ_ATTRS]]

; CHECK: declare double @nexttoward(double, x86_fp80) [[ERRNO_ATTRS]]
; CHECK: declare float @nexttowardf(float, x86_fp80) [[ERRNO_ATTRS]]
; CHECK: declare x86_fp80 @nexttowardl(x86_fp80, x86_fp80) [[ERRNO_ATTRS]]

; CHECK: declare double @remainder(double, double) [[ERRNO_ATTRS]]
; CHECK: declare float @remainderf(float, float) [[ERRNO_ATTRS]]
; CHECK: declare x86_fp80 @remainderl(x86_fp80, x86_fp80) [[ERRNO_ATTRS]]

; CHECK: declare double @remquo(double, double, ptr noalias nonnull writeonly align 4 dereferenceable(4)) [[ARGMEM_ERRNO_WRITE_ATTRS:#[0-9]+]]
; CHECK: declare float @remquof(float, float, ptr noalias nonnull writeonly align 4 dereferenceable(4)) [[ARGMEM_ERRNO_WRITE_ATTRS]]
; CHECK: declare x86_fp80 @remquol(x86_fp80, x86_fp80, ptr noalias nonnull writeonly align 4 dereferenceable(4)) [[ARGMEM_ERRNO_WRITE_ATTRS]]

; CHECK: declare double @scalbln(double, i64) [[ERRNO_ATTRS]]
; CHECK: declare float @scalblnf(float, i64) [[ERRNO_ATTRS]]
; CHECK: declare x86_fp80 @scalblnl(x86_fp80, i64) [[ERRNO_ATTRS]]

; CHECK: declare double @scalbn(double, i32) [[ERRNO_ATTRS]]
; CHECK: declare float @scalbnf(float, i32) [[ERRNO_ATTRS]]
; CHECK: declare x86_fp80 @scalbnl(x86_fp80, i32) [[ERRNO_ATTRS]]

; CHECK: declare nofpclass(ninf nsub nnorm) double @sqrt(double) [[ERRNO_ATTRS]]

; CHECK: declare nofpclass(ninf nsub nnorm) float @sqrtf(float) [[ERRNO_ATTRS]]

; CHECK: declare double @tgamma(double) [[ERRNO_ATTRS]]
; CHECK: declare float @tgammaf(float) [[ERRNO_ATTRS]]
; CHECK: declare x86_fp80 @tgammal(x86_fp80) [[ERRNO_ATTRS]]

; CHECK: declare x86_fp80 @truncl(x86_fp80) [[NOMEM_ATTRS]]

; CHECK: attributes [[NOMEM_ATTRS]] = { nocallback nofree nosync nounwind willreturn memory(none) }
; CHECK: attributes [[ATOMIC_ATTRS]] = { nocallback nofree nounwind willreturn }
; CHECK: attributes [[ERRNO_ATTRS]] = { nocallback nofree nosync nounwind willreturn memory(errnomem: write) }
; CHECK: attributes [[ARGMEM_ATTRS]] = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
; CHECK: attributes [[ARGMEM_WRITE_ATTRS]] = { nocallback nofree nosync nounwind willreturn memory(argmem: write) }
; CHECK: attributes [[CALLOC_ATTRS]] = { nofree nounwind willreturn allockind("alloc,zeroed") allocsize(0,1) memory(inaccessiblemem: readwrite, errnomem: readwrite) "alloc-family"="malloc" }
; CHECK: attributes [[FREE_ATTRS]] = { nounwind willreturn allockind("free") memory(argmem: readwrite, inaccessiblemem: readwrite) "alloc-family"="malloc" }
; CHECK: attributes [[MALLOC_ATTRS]] = { nofree nounwind willreturn allockind("alloc,uninitialized") allocsize(0) memory(inaccessiblemem: readwrite, errnomem: readwrite) "alloc-family"="malloc" }
; CHECK: attributes [[ARGMEM_READ_ATTRS]] = { nocallback nofree nosync nounwind willreturn memory(argmem: read) }
; CHECK: attributes [[ARGMEM_ERRNO_WRITE_ATTRS]] = { nocallback nofree nosync nounwind willreturn memory(argmem: write, errnomem: write) }
