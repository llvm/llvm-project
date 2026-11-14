; RUN: llc -O3 -mattr=+avx512f -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -O3 -mcpu=znver4 -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -O3 -mcpu=skylake-avx512 -verify-machineinstrs < %s | FileCheck %s
;
; The compare's memory operand overlaps the store at byte offset 72.
; It must observe the old value. When the narrow load is folded into an
; overlapping wide load, its outgoing memory dependencies must be preserved.
;
; CHECK-LABEL: changed:
; CHECK:       movq 80(%rdi), [[VALUE:%[a-z0-9]+]]
; CHECK-NOT:   movq {{.*}}, 72(%rdi)
; CHECK:       vpcmpneqd 16(%rdi), {{%zmm[0-9]+}}, {{%k[0-9]+}}
; CHECK:       movq [[VALUE]], 72(%rdi)
; CHECK:       retq

target triple = "x86_64-unknown-linux-gnu"

define i1 @changed(ptr dereferenceable(88) %p) nofree {
  %old = load <10 x i64>, ptr %p, align 8
  %before = shufflevector <10 x i64> %old, <10 x i64> poison, <8 x i32> <i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9>
  %input = getelementptr i64, ptr %p, i64 10
  %value = load i64, ptr %input, align 8
  %after = insertelement <8 x i64> zeroinitializer, i64 %value, i64 7
  %dst = getelementptr i64, ptr %p, i64 9
  store i64 %value, ptr %dst, align 8
  %cmp = icmp ne <8 x i64> %before, %after
  %bits = bitcast <8 x i1> %cmp to i8
  %result = icmp ne i8 %bits, 0
  ret i1 %result
}
