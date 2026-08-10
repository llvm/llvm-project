; RUN: llc -O2 -no-integrated-as < %s | FileCheck %s

; XCore default subtarget does not support 8-byte alignment on stack.
; XFAIL: target=xcore{{.*}}

@G = common global i32 0, align 4

define i32 @foo(ptr %p) nounwind uwtable {
entry:
  %p.addr = alloca ptr, align 8
  %rv = alloca i32, align 4
  store ptr %p, ptr %p.addr, align 8
  store i32 0, ptr @G, align 4
  %0 = load ptr, ptr %p.addr, align 8
; CHECK: blah
  %1 = call i32 asm "blah", "=r,r,~{memory}"(ptr %0) nounwind
; CHECK: {{[^[:alnum:]]}}G{{[^[:alnum:]]}}
  store i32 %1, ptr %rv, align 4
  %2 = load i32, ptr %rv, align 4
  %3 = load i32, ptr @G, align 4
  %add = add nsw i32 %2, %3
  ret i32 %add
}

