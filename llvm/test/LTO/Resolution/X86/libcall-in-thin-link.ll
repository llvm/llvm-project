;; If a libcall was extracted in a thin link, it can be used even if not
;; present in the current TU.

; RUN: rm -rf %t && split-file %s %t && cd %t
; RUN: opt foo.ll -o foo.o -module-summary -mtriple x86_64-unknown-linux-musl
; RUN: opt bcmp.ll -o bcmp.o -module-summary -mtriple x86_64-unknown-linux-musl
; RUN: llvm-lto2 run -o lto.o \
; RUN:   -r foo.o,foo,plx \
; RUN:   -r foo.o,memcmp,x \
; RUN:   -r bcmp.o,bcmp,pl \
; RUN:   -r bcmp.o,bcmp_impl,x foo.o bcmp.o -save-temps
; RUN: llvm-dis lto.o.1.4.opt.bc -o - | FileCheck %s

;--- foo.ll
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

;--- bcmp.ll
define i32 @bcmp(ptr %0, ptr %1, i64 %2) noinline {
  %r = call i32 @bcmp_impl(ptr %0, ptr %1, i64 %2)
  ret i32 %r
}

declare i32 @bcmp_impl(ptr, ptr, i64)

