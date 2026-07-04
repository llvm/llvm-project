; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+sse2,+bmi,+tbm | FileCheck %s --check-prefixes=X64

; Tests for ValueTracking known-bits handling of BEXTR/BEXTRI intrinsics.

declare i32 @llvm.x86.bmi.bextr.32(i32, i32)
declare i64 @llvm.x86.bmi.bextr.64(i64, i64)
declare i32 @llvm.x86.tbm.bextri.u32(i32, i32) nounwind readnone
declare i64 @llvm.x86.tbm.bextri.u64(i64, i64) nounwind readnone

define i32 @vt_bextr_zero32(i32 %x) nounwind {
; X86-LABEL: vt_bextr_zero32:
; X86:       # %bb.0:
; X86-NEXT:    xorl %eax, %eax
; X86-NEXT:    retl
;
; X64-LABEL: vt_bextr_zero32:
; X64:    xorl %eax, %eax
; X64:    retq
entry:
  %1 = tail call i32 @llvm.x86.bmi.bextr.32(i32 %x, i32 0)
  ret i32 %1
}

define i64 @vt_bextr_zero64(i64 %x) nounwind {
; X86-LABEL: vt_bextr_zero64:
; X86-NEXT:    xorl %eax, %eax
; X86-NEXT:    retl
;
; X64-LABEL: vt_bextr_zero64:
; X64:    xorl %eax, %eax
; X64:    retq
entry:
  %1 = tail call i64 @llvm.x86.bmi.bextr.64(i64 %x, i64 0)
  ret i64 %1
}

define i32 @vt_bextr_extract32(i32 %x) nounwind {
; X86-LABEL: vt_bextr_extract32:
; X86:       # %bb.0:
; X86-NEXT:    bextrl $2052, {{[0-9]+}}(%esp), %eax # imm = 0x804
; X86-NEXT:    retl
;
; X64-LABEL: vt_bextr_extract32:
; X64:    movl $2052, %eax # imm = 0x804
; X64:    bextrl
; X64:    retq
entry:
  %1 = tail call i32 @llvm.x86.bmi.bextr.32(i32 %x, i32 2052)
  ret i32 %1
}

define i64 @vt_bextr_extract64(i64 %x) nounwind {
; X86-LABEL: vt_bextr_extract64:
; X86:       # %bb.0:
; X86-NEXT:    bextrq %rsi, %rdi, %rax
; X86-NEXT:    retl
;
; X64-LABEL: vt_bextr_extract64:
; X64:    movl $1028, %eax # imm = 0x404
; X64:    bextrq
; X64:    retq
entry:
  %1 = tail call i64 @llvm.x86.bmi.bextr.64(i64 %x, i64 1028)
  ret i64 %1
}

define i32 @vt_bextri_zero32(i32 %x) nounwind {
; X86-LABEL: vt_bextri_zero32:
; X86-NEXT:    xorl %eax, %eax
; X86-NEXT:    retl
;
; X64-LABEL: vt_bextri_zero32:
; X64:    xorl %eax, %eax
; X64:    retq
entry:
  %1 = tail call i32 @llvm.x86.tbm.bextri.u32(i32 %x, i32 0)
  ret i32 %1
}

define i64 @vt_bextri_zero64(i64 %x) nounwind {
; X86-LABEL: vt_bextri_zero64:
; X86-NEXT:    xorl %eax, %eax
; X86-NEXT:    retl
;
; X64-LABEL: vt_bextri_zero64:
; X64:    xorl %eax, %eax
; X64:    retq
entry:
  %1 = tail call i64 @llvm.x86.tbm.bextri.u64(i64 %x, i64 0)
  ret i64 %1
}

define i32 @vt_bextri_extract32(i32 %x) nounwind {
; X86-LABEL: vt_bextri_extract32:
; X86-NEXT:    bextrl $2052, {{[0-9]+}}(%esp), %eax # imm = 0x804
; X86-NEXT:    retl
;
; X64-LABEL: vt_bextri_extract32:
; X64:    bextrl $2052, %edi, %eax # imm = 0x804
; X64:    retq
entry:
  %1 = tail call i32 @llvm.x86.tbm.bextri.u32(i32 %x, i32 2052)
  ret i32 %1
}

define i64 @vt_bextri_extract64(i64 %x) nounwind {
; X86-LABEL: vt_bextri_extract64:
; X86-NEXT:    bextrq %rsi, %rdi, %rax
; X86-NEXT:    retl
;
; X64-LABEL: vt_bextri_extract64:
; X64:    bextrq
; X64:    retq
entry:
  %1 = tail call i64 @llvm.x86.tbm.bextri.u64(i64 %x, i64 1028)
  ret i64 %1
}
