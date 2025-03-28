; RUN: llc -mcpu=ppc -mtriple=powerpc-ibm-aix-xcoff %s -filetype=obj -o %t
; RUN: llvm-objdump %t -r -d --symbolize-operands --no-show-raw-insn \
; RUN:   | FileCheck %s

; CHECK-LABEL: <.a>:
;; No <L0> should appear
; CHECK-NEXT:       0:      	mflr 0
; CHECK-NEXT:       4:      	stwu 1, -64(1)
; CHECK-NEXT:       8:      	lwz 3, 0(2)
; CHECK-NEXT:0000000a:  R_TOC        var
; CHECK-NEXT:       c:      	stw 0, 72(1)
; CHECK-NEXT:      10:      	lwz 3, 0(3)
; CHECK-NEXT:      14:      	bl 0x4c <.b>
; CHECK-NEXT:      18:      	nop
; CHECK-NEXT:      1c:      	li 3, 1
; CHECK-NEXT:      20:      	bl 0x0 <.c>
; CHECK-NEXT:00000020:  R_RBR        .c

; CHECK-LABEL: <.b>:
; CHECK-NEXT:      4c:      	mflr 0
; CHECK-NEXT:      50:      	stwu 1, -64(1)
; CHECK-NEXT:      54:      	cmplwi	3, 1
; CHECK-NEXT:      58:      	stw 0, 72(1)
; CHECK-NEXT:      5c:      	stw 3, 60(1)
; CHECK-NEXT:      60:      	bf	2, 0x6c <L0>
; CHECK-NEXT:      64:      	bl 0x0 <.a>
; CHECK-NEXT:      68:      	nop
; CHECK-NEXT:<L0>:
; CHECK-NEXT:      6c:      	li 3, 2
; CHECK-NEXT:      70:      	bl 0x0 <.c>
; CHECK-NEXT:00000070:  R_RBR        .c

target triple = "powerpc-ibm-aix7.2.0.0"

@var = external global i32, align 4

; Function Attrs: noinline nounwind optnone
define i32 @a() {
entry:
  %0 = load i32, ptr @var, align 4
  %call = call i32 @b(i32 noundef %0)
  %call1 = call i32 @c(i32 noundef 1)
  ret i32 %call1
}

; Function Attrs: noinline nounwind optnone
define i32 @b(i32 noundef %x) {
entry:
  %x.addr = alloca i32, align 4
  store i32 %x, ptr %x.addr, align 4
  %0 = load i32, ptr %x.addr, align 4
  %cmp = icmp eq i32 %0, 1
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %call = call i32 @a()
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  %call1 = call i32 @c(i32 noundef 2)
  ret i32 %call1
}

declare i32 @c(i32 noundef)

