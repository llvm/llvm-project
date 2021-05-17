; RUN: llc -O2 -no-integrated-as < %s | FileCheck %s

; XCore default subtarget does not support 8-byte alignment on stack.
; XFAIL: xcore

@G = common global i32 0, align 4

define i32 @foo(i8* %p) nounwind uwtable {
entry:
  %p.addr = alloca i8*, align 8
  %rv = alloca i32, align 4
  store i8* %p, i8** %p.addr, align 8
  store i32 0, i32* @G, align 4
  %0 = load i8*, i8** %p.addr, align 8
; CHECK: blah
  %1 = call i32 asm "blah", "=r,r,~{memory}"(i8* %0) nounwind
; CHECK: @G
  store i32 %1, i32* %rv, align 4
  %2 = load i32, i32* %rv, align 4
  %3 = load i32, i32* @G, align 4
  %add = add nsw i32 %2, %3
  ret i32 %add
}

