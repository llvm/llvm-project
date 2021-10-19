; RUN: llc -O0 -mtriple=i386-unknown-linux-gnu -relocation-model=pic < %s | FileCheck %s

; Unbind the ebx with GOT address in regcall calling convention, or the following
; case will failed in register allocation by no register can be used.

;#define REGCALL __attribute__((regcall))
;int REGCALL func (int i1, int i2, int i3, int i4, int i5);
;int (REGCALL *fptr) (int, int, int, int, int) = func;
;int test() {
;    return fptr(1,2,3,4,5);
;}

@fptr = global i32 (i32, i32, i32, i32, i32)* @__regcall3__func, align 4

declare x86_regcallcc i32 @__regcall3__func(i32 inreg, i32 inreg, i32 inreg, i32 inreg, i32 inreg)

; Function Attrs: noinline nounwind optnone
define i32 @test() {
; CHECK-LABEL: test:
; CHECK:       .L0$pb:
; CHECK-NEXT:    popl %eax
; CHECK:       .Ltmp0:
; CHECK-NEXT:    addl    $_GLOBAL_OFFSET_TABLE_+(.Ltmp0-.L0$pb), %eax
; CHECK-NEXT:    movl    fptr@GOT(%eax), %eax
; CHECK-NEXT:    movl    (%eax), %ebx
; CHECK-NEXT:    movl    $1, %eax
; CHECK-NEXT:    movl    $2, %ecx
; CHECK-NEXT:    movl    $3, %edx
; CHECK-NEXT:    movl    $4, %edi
; CHECK-NEXT:    movl    $5, %esi
; CHECK-NEXT:    calll   *%ebx

entry:
  %0 = load i32 (i32, i32, i32, i32, i32)*, i32 (i32, i32, i32, i32, i32)** @fptr, align 4
  %call = call x86_regcallcc i32 %0(i32 inreg 1, i32 inreg 2, i32 inreg 3, i32 inreg 4, i32 inreg 5)
  ret i32 %call
}
