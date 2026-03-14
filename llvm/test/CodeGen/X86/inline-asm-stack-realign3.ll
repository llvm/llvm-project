; RUN: llc -mtriple=i686-- -no-integrated-as < %s | FileCheck %s

declare void @bar(ptr %junk)

define i32 @foo(i1 %cond) {
entry:
  %r = alloca i32, align 128
  store i32 -1, ptr %r, align 128
  br i1 %cond, label %doit, label %skip

doit:
  call void asm sideeffect "xor %ecx, %ecx\0A\09mov %ecx, $0", "=*m,~{ecx},~{flags}"(ptr elementtype(i32) %r)
  %junk = alloca i32
  call void @bar(ptr %junk)
  br label %skip

skip:
  %0 = load i32, ptr %r, align 128
  ret i32 %0
}

; CHECK-LABEL: foo:
; CHECK: pushl %ebp
; CHECK: andl $-128, %esp
; CHECK: xor %ecx, %ecx
; CHECK-NEXT: mov %ecx, (%esi)
; CHECK: movl (%esi), %eax
; CHECK: popl %ebp
; CHECK: ret
