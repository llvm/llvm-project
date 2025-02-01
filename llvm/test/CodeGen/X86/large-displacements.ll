; RUN: not llc < %s -mtriple=i686 2>&1 | FileCheck %s -check-prefix=i686
; RUN: llc < %s -mtriple=x86_64 | FileCheck %s -check-prefix=x86_64

define dso_local i32 @main() #0 {
; i686: error: <unknown>:0:0: requesting 64-bit offset in 32-bit immediate: main
;
; x86_64-LABEL: main:
; x86_64:       # %bb.0: # %entry
; x86_64-NEXT:    movl $4294967176, %eax # imm = 0xFFFFFF88
; x86_64-NEXT:    subq %rax, %rsp
; x86_64-NEXT:    .cfi_def_cfa_offset 4294967184
; x86_64-NEXT:    movb $32, -1073741994(%rsp)
; x86_64-NEXT:    movb $33, 2147483478(%rsp)
; x86_64-NEXT:    movb $34, 1073741654(%rsp)
; x86_64-NEXT:    movb $35, -170(%rsp)
; x86_64-NEXT:    xorl %eax, %eax
; x86_64-NEXT:    movl $4294967176, %ecx # imm = 0xFFFFFF88
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
