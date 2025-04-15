;; This test comes from a real world scenario in LTO, where the
;; definition of bcmp is deleted because it has no uses, but later instcombine
;; re-introduces a call to bcmp() as part of SimplifyLibCalls.

; RUN: opt %s -o %t -module-summary -mtriple x86_64-unknown-linux-musl
; RUN: llvm-lto2 run -o %t2 \
; RUN:   -r %t,foo,plx \
; RUN:   -r %t,memcmp,x \
; RUN:   -r %t,bcmp,pl %t -save-temps
; RUN: llvm-dis %t2.1.2.internalize.bc -o - \
; RUN:   | FileCheck %s

define i1 @foo(ptr %0, [2 x i32] %1) {
  ; CHECK-LABEL: define{{.*}}i1 @foo
  ; CHECK-NEXT: %size = extractvalue [2 x i32] %1, 1
  ; CHECK-NEXT: %cmp = {{.*}}call i32 @memcmp
  ; CHECK-NEXT: %eq = icmp eq i32 %cmp, 0
  ; CHECK-NEXT: ret i1 %eq

  %size = extractvalue [2 x i32] %1, 1
  %cmp = call i32 @memcmp(ptr %0, ptr null, i32 %size)
  %eq = icmp eq i32 %cmp, 0
  ret i1 %eq
}

; CHECK: declare i32 @memcmp(ptr, ptr, i32)
declare i32 @memcmp(ptr, ptr, i32)

;; Ensure bcmp is removed from module. Follow up patches can address this.
; CHECK-NOT: declare{{.*}}i32 @bcmp
; CHECK-NOT: define{{.*}}i32 @bcmp
define i32 @bcmp(ptr %0, ptr %1, i32 %2) {
  ret i32 0
}

