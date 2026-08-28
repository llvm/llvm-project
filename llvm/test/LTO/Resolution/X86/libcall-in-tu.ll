;; This test comes from a real world scenario in LTO, where the definition of
;; bcmp was deleted because it has no uses, but later instcombine re-introduced
;; a call to bcmp() as part of SimplifyLibCalls. Such deletions must not be
;; allowed.

; RUN: opt %s -o %t.o -mtriple x86_64-unknown-linux-musl
; RUN: llvm-lto2 run -o %t.lto.o \
; RUN:   -r %t.o,foo,plx \
; RUN:   -r %t.o,memcmp,x \
; RUN:   -r %t.o,bcmp,pl \
; RUN:   -r %t.o,bcmp_impl,x %t.o -save-temps
; RUN: llvm-dis %t.lto.o.0.4.opt.bc -o - | FileCheck %s

define i1 @foo(ptr %0, ptr %1, i64 %2) {
  ; CHECK-LABEL: define{{.*}}i1 @foo
  ; CHECK-NEXT: %bcmp = {{.*}}call i32 @bcmp
  ; CHECK-NEXT: %eq = icmp eq i32 %bcmp, 0
  ; CHECK-NEXT: ret i1 %eq

  %cmp = call i32 @memcmp(ptr %0, ptr %1, i64 %2)
  %eq = icmp eq i32 %cmp, 0
  ret i1 %eq
}

declare i32 @memcmp(ptr, ptr, i64)
declare i32 @bcmp_impl(ptr, ptr, i64)

;; Ensure bcmp is not removed from module because it is external.
; CHECK: define dso_local i32 @bcmp
define i32 @bcmp(ptr %0, ptr %1, i64 %2) noinline {
  %r = call i32 @bcmp_impl(ptr %0, ptr %1, i64 %2)
  ret i32 %r
}

