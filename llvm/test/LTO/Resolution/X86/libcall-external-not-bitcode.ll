; RUN: opt %s -o %t -module-summary -mtriple x86_64-unknown-linux-musl
; RUN: llvm-lto2 run -o %t2 \
; RUN:   -r %t,foo,plx \
; RUN:   -r %t,memcmp,x \
; RUN:   -r %t,bcmp,pl %t -save-temps
; RUN: llvm-dis %t2.1.4.opt.bc -o - | FileCheck %s

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
declare i32 @bcmp(ptr, ptr, i64)
