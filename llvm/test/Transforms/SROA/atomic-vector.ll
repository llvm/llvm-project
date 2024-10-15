; RUN: opt < %s -passes='sroa' -S 2>&1 | FileCheck %s --check-prefix=ERR
; RUN: opt < %s -passes='sroa' -S | FileCheck %s

define float @atomic_vector() {
; ERR-NOT: atomic load operand must have integer, pointer, or floating point type!
; ERR-NOT:   <1 x float>  {{%.*}} = load atomic volatile <1 x float>, ptr {{%.*}} acquire, align 4
; CHECK:      %1 = alloca <1 x float>, align 4
; CHECK-NEXT: store <1 x float> undef, ptr %1, align 4
; CHECK-NEXT: %2 = load atomic volatile float, ptr %1 acquire, align 4
; CHECK-NEXT: ret float %2
  %1 = alloca <1 x float>
  %2 = alloca <1 x float>
  %3 = alloca ptr
  call void @llvm.memcpy.p0.p0.i64(ptr %2, ptr %1, i64 4, i1 false)
  store ptr %2, ptr %3
  %4 = load ptr, ptr %3
  %5 = load atomic volatile float, ptr %4 acquire, align 4
  ret float %5
}
