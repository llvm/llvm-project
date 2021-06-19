; RUN: llc -mtriple=x86_64-unknown-unknown < %s | FileCheck %s

%struct.__va_list_tag = type { i32, i32, i8*, i8* }

declare void @llvm.va_end(i8*) #0
declare void @llvm.va_start(i8*) #10

define void @hardf(i8* %fmt, ...) #1 {
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
  %arraydecay = getelementptr inbounds [1 x %struct.__va_list_tag], [1 x %struct.__va_list_tag]* %va, i64 0, i64 0
  %a = bitcast %struct.__va_list_tag* %arraydecay to i8*
  call void @llvm.va_start(i8* %a)
  call void @llvm.va_end(i8* nonnull %a)
  ret void
}

define void @softf(i8* %fmt, ...) #2 {
; CHECK-LABEL: softf
; For software floating point, floats are passed in general
; purpose registers, so no need to spill XMM registers.
; CHECK-NOT: %xmm
; CHECK: retq
  %va = alloca [1 x %struct.__va_list_tag], align 16
  %arraydecay = getelementptr inbounds [1 x %struct.__va_list_tag], [1 x %struct.__va_list_tag]* %va, i64 0, i64 0
  %a = bitcast %struct.__va_list_tag* %arraydecay to i8*
  call void @llvm.va_start(i8* %a)
  call void @llvm.va_end(i8* nonnull %a)
  ret void
}

define void @noimplf(i8* %fmt, ...) #3 {
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
  %arraydecay = getelementptr inbounds [1 x %struct.__va_list_tag], [1 x %struct.__va_list_tag]* %va, i64 0, i64 0
  %a = bitcast %struct.__va_list_tag* %arraydecay to i8*
  call void @llvm.va_start(i8* %a)
  call void @llvm.va_end(i8* nonnull %a)
  ret void
}

define void @noimplsoftf(i8* %fmt, ...) #4 {
; CHECK-LABEL: noimplsoftf
; Combining noimplicitfloat and use-soft-float should not assert (PR48528).
; CHECK-NOT: %xmm
; CHECK: retq
  %va = alloca [1 x %struct.__va_list_tag], align 16
  %arraydecay = getelementptr inbounds [1 x %struct.__va_list_tag], [1 x %struct.__va_list_tag]* %va, i64 0, i64 0
  %a = bitcast %struct.__va_list_tag* %arraydecay to i8*
  call void @llvm.va_start(i8* %a)
  call void @llvm.va_end(i8* nonnull %a)
  ret void
}

attributes #0 = { nounwind }
attributes #1 = { nounwind uwtable }
attributes #2 = { nounwind uwtable "use-soft-float"="true" }
attributes #3 = { noimplicitfloat nounwind uwtable }
attributes #4 = { noimplicitfloat nounwind uwtable "use-soft-float"="true" }
