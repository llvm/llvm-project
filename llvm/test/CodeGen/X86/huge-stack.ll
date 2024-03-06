; RUN: llc -O0 -mtriple=x86_64 < %s | FileCheck %s --check-prefix=CHECK
%large = type [4294967295 x i8]

define void @foo() unnamed_addr #0 {
  %1 = alloca %large, align 1
  %2 = alloca %large, align 1
  %3 = getelementptr inbounds %large, %large* %1, i64 0, i64 0
  store i8 42, i8* %3, align 1
  %4 = getelementptr inbounds %large, %large* %2, i64 0, i64 0
  store i8 43, i8* %4, align 1
  ret void
}

; CHECK-LABEL: foo:
; CHECK:      movabsq $8589934462, %rax
; CHECK-NEXT: subq    %rax, %rsp
; CHECK-NEXT: .cfi_def_cfa_offset 8589934470
; CHECK-NEXT: movb $42, 4294967167(%rsp)
; CHECK-NEXT: movb $43, -128(%rsp)
; CHECK-NEXT: movabsq $8589934462, %rax
; CHECK-NEXT: addq %rax, %rsp
; CHECK-NEXT: .cfi_def_cfa_offset 8
