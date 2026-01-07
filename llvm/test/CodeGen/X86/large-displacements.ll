; RUN: not llc < %s -mtriple=i686 -filetype=null -verify-machineinstrs 2>&1 | FileCheck %s -check-prefix=ERR-i686
; RUN: llc < %s -mtriple=x86_64 -verify-machineinstrs | FileCheck %s -check-prefix=x86_64

; Regression test for #121932, #113856, #106352, #69365, #25051 which are caused by
; an incorrectly written assertion for 64-bit offsets when compiling for 32-bit X86.

define i32 @main() #0 {
; ERR-i686: error: <unknown>:0:0: 64-bit offset calculated but target is 32-bit
;
; x86_64-LABEL: main:
; x86_64:       # %bb.0: # %entry
; x86_64-NEXT:    movl $4294967192, %eax # imm = 0xFFFFFF98
; x86_64-NEXT:    subq %rax, %rsp
; x86_64-NEXT:    .cfi_def_cfa_offset 4294967200
; x86_64-NEXT:    movabsq $3221225318, %rax # imm = 0xBFFFFF66
; x86_64-NEXT:    movb $32, (%rsp,%rax)
; x86_64-NEXT:    movb $33, 2147483494(%rsp)
; x86_64-NEXT:    movb $34, 1073741670(%rsp)
; x86_64-NEXT:    movb $35, -154(%rsp)
; x86_64-NEXT:    xorl %eax, %eax
; x86_64-NEXT:    movl $4294967192, %ecx # imm = 0xFFFFFF98
; x86_64-NEXT:    addq %rcx, %rsp
; x86_64-NEXT:    .cfi_def_cfa_offset 8
; x86_64-NEXT:    retq
entry:
  %a = alloca [1073741824 x i8], align 16
  %b = alloca [1073741824 x i8], align 16
  %c = alloca [1073741824 x i8], align 16
  %d = alloca [1073741824 x i8], align 16

  %arrayida = getelementptr inbounds [1073741824 x i8], ptr %a, i64 0, i64 -42
  %arrayidb = getelementptr inbounds [1073741824 x i8], ptr %b, i64 0, i64 -42
  %arrayidc = getelementptr inbounds [1073741824 x i8], ptr %c, i64 0, i64 -42
  %arrayidd = getelementptr inbounds [1073741824 x i8], ptr %d, i64 0, i64 -42

  store i8 32, ptr %arrayida, align 2
  store i8 33, ptr %arrayidb, align 2
  store i8 34, ptr %arrayidc, align 2
  store i8 35, ptr %arrayidd, align 2

  ret i32 0
}

; Same test as above but for an anonymous function.
define i32 @0() #0 {
; ERR-i686: error: <unknown>:0:0: 64-bit offset calculated but target is 32-bit
;
; x86_64-LABEL: __unnamed_1:
; x86_64:       # %bb.0: # %entry
; x86_64-NEXT:    movl $4294967192, %eax # imm = 0xFFFFFF98
; x86_64-NEXT:    subq %rax, %rsp
; x86_64-NEXT:    .cfi_def_cfa_offset 4294967200
; x86_64-NEXT:    movabsq $3221225318, %rax # imm = 0xBFFFFF66
; x86_64-NEXT:    movb $32, (%rsp,%rax)
; x86_64-NEXT:    movb $33, 2147483494(%rsp)
; x86_64-NEXT:    movb $34, 1073741670(%rsp)
; x86_64-NEXT:    movb $35, -154(%rsp)
; x86_64-NEXT:    xorl %eax, %eax
; x86_64-NEXT:    movl $4294967192, %ecx # imm = 0xFFFFFF98
; x86_64-NEXT:    addq %rcx, %rsp
; x86_64-NEXT:    .cfi_def_cfa_offset 8
; x86_64-NEXT:    retq
entry:
  %a = alloca [1073741824 x i8], align 16
  %b = alloca [1073741824 x i8], align 16
  %c = alloca [1073741824 x i8], align 16
  %d = alloca [1073741824 x i8], align 16

  %arrayida = getelementptr inbounds [1073741824 x i8], ptr %a, i64 0, i64 -42
  %arrayidb = getelementptr inbounds [1073741824 x i8], ptr %b, i64 0, i64 -42
  %arrayidc = getelementptr inbounds [1073741824 x i8], ptr %c, i64 0, i64 -42
  %arrayidd = getelementptr inbounds [1073741824 x i8], ptr %d, i64 0, i64 -42

  store i8 32, ptr %arrayida, align 2
  store i8 33, ptr %arrayidb, align 2
  store i8 34, ptr %arrayidc, align 2
  store i8 35, ptr %arrayidd, align 2

  ret i32 0
}

attributes #0 = { optnone noinline }
