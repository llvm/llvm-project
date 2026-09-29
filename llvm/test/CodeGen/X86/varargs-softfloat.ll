; RUN: llc -mtriple=x86_64-unknown-unknown < %s | FileCheck %s

%struct.__va_list_tag = type { i32, i32, ptr, ptr }

declare void @llvm.va_end(ptr) #0
declare void @llvm.va_start(ptr) #10

define void @hardf(ptr %fmt, ...) #1 {
; CHECK-LABEL: hardf
; When using XMM registers to pass floating-point parameters,
; we need to spill those for va_start.
; CHECK: testb %al, %al
; CHECK: movaps  %xmm0, {{.*}}%rsp
; CHECK: movaps  %xmm1, {{.*}}%rsp
; CHECK: movaps  %xmm2, {{.*}}%rsp
; CHECK: movaps  %xmm3, {{.*}}%rsp
; CHECK: movaps  %xmm4, {{.*}}%rsp
; CHECK: movaps  %xmm5, {{.*}}%rsp
; CHECK: movaps  %xmm6, {{.*}}%rsp
; CHECK: movaps  %xmm7, {{.*}}%rsp
  %va = alloca [1 x %struct.__va_list_tag], align 16
  call void @llvm.va_start(ptr %va)
  call void @llvm.va_end(ptr nonnull %va)
  ret void
}

define void @softf(ptr %fmt, ...) #2 {
; CHECK-LABEL: softf
; For software floating point, floats are passed in general
; purpose registers, so no need to spill XMM registers.
; CHECK-NOT: %xmm
; CHECK: retq
  %va = alloca [1 x %struct.__va_list_tag], align 16
  call void @llvm.va_start(ptr %va)
  call void @llvm.va_end(ptr nonnull %va)
  ret void
}

define void @noimplf(ptr %fmt, ...) #3 {
; CHECK-LABEL: noimplf
; Even with noimplicitfloat, when using the hardware float API, we
; need to emit code to spill the XMM registers (PR36507).
; CHECK: testb %al, %al
; CHECK: movaps  %xmm0, {{.*}}%rsp
; CHECK: movaps  %xmm1, {{.*}}%rsp
; CHECK: movaps  %xmm2, {{.*}}%rsp
; CHECK: movaps  %xmm3, {{.*}}%rsp
; CHECK: movaps  %xmm4, {{.*}}%rsp
; CHECK: movaps  %xmm5, {{.*}}%rsp
; CHECK: movaps  %xmm6, {{.*}}%rsp
; CHECK: movaps  %xmm7, {{.*}}%rsp
  %va = alloca [1 x %struct.__va_list_tag], align 16
  call void @llvm.va_start(ptr %va)
  call void @llvm.va_end(ptr nonnull %va)
  ret void
}

define void @noimplsoftf(ptr %fmt, ...) #4 {
; CHECK-LABEL: noimplsoftf
; Combining noimplicitfloat and use-soft-float should not assert (PR48528).
; CHECK-NOT: %xmm
; CHECK: retq
  %va = alloca [1 x %struct.__va_list_tag], align 16
  call void @llvm.va_start(ptr %va)
  call void @llvm.va_end(ptr nonnull %va)
  ret void
}

attributes #0 = { nounwind }
attributes #1 = { nounwind uwtable }
attributes #2 = { nounwind uwtable "use-soft-float"="true" }
attributes #3 = { noimplicitfloat nounwind uwtable }
attributes #4 = { noimplicitfloat nounwind uwtable "use-soft-float"="true" }
