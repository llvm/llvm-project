; Tests that basic block sections avoids inserting an unconditional branch when a basic block
; already has an unconditional branch to its fallthrough block.
; RUN: llc < %s -mtriple=x86_64 -basic-block-sections=all -O0 | FileCheck %s
; This test case is generated from code:
; int
; mapping (int len)
; {
;   switch (len)
;   {
;     case 7: return 333;
;     default:
;       goto unknown;
;   }
; unknown:
;   return 0;
; }
; clang -O0 -fbasic-block-sections=all test.c

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @mapping(i32 noundef %len) {
entry:
  %retval = alloca i32, align 4
  %len.addr = alloca i32, align 4
  store i32 %len, ptr %len.addr, align 4
  %0 = load i32, ptr %len.addr, align 4
  switch i32 %0, label %sw.default [
    i32 7, label %sw.bb
  ]

sw.bb:                                            ; preds = %entry
  store i32 333, ptr %retval, align 4
  br label %return

sw.default:                                       ; preds = %entry
  br label %unknown

unknown:                                          ; preds = %sw.default
  store i32 0, ptr %retval, align 4
  br label %return

return:                                           ; preds = %unknown, %sw.bb
  %1 = load i32, ptr %retval, align 4
  ret i32 %1
}

; CHECK: mapping:
; CHECK-NEXT: .cfi_startproc
; CHECK-NEXT: # %bb.0: # %entry
; CHECK-NEXT: movl {{.*}}
; CHECK-NEXT: movl {{.*}}
; CHECK-NEXT: subl {{.*}}
; CHECK-NEXT: jne	mapping.__part.2
; CHECK-NEXT: jmp	mapping.__part.1
; CHECK-NOT:  jmp
; CHECK: mapping.__part.1:
; CHECK: mapping.__part.2:
