; RUN: llc -mtriple=bpf -mcpu=generic -mattr=+has-i128-direct-return < %s | FileCheck %s

; Source code:
; void test(__int128 *a) {
;     __int128 tmp = __atomic_load_n(a, __ATOMIC_RELAXED);
;     __atomic_store_n(a, tmp, __ATOMIC_RELAXED);
; }
; 
; Compile with:
; 	clang -target bpf -O2 -S -emit-llvm test.c

define void @test(ptr %a) nounwind {
; CHECK-LABEL: test:
; CHECK: r6 = r1
; CHECK-NEXT: r2 = 0
; CHECK-NEXT: call __atomic_load_16
; CHECK-NEXT: r3 = r1
; CHECK-NEXT: r1 = r6
; CHECK-NEXT: r2 = r0
; CHECK-NEXT: r4 = 0
; CHECK-NEXT: call __atomic_store_16
  %1 = load atomic i128, ptr %a monotonic, align 16
  store atomic i128 %1, ptr %a monotonic, align 16
  ret void
}
