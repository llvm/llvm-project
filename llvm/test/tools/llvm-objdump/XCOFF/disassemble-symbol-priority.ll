; RUN: llc -mtriple=powerpc-ibm-aix-xcoff %s -filetype=obj -o %t
; RUN: llvm-objdump %t -d --no-show-raw-insn | FileCheck %s

; CHECK: Disassembly of section .text:
; CHECK: 00000000 <.foo3>:
; CHECK: 00000020 <.foo4>:
; CHECK: 00000040 <.foo>:
; CHECK: 00000060 <.foo2>:

define dso_local signext i32 @foo(i32 noundef signext %a) #0 section "explicit_sec" {
entry:
  ret i32 %a
}

define dso_local signext i32 @foo2(i32 noundef signext %a) #0 section "explicit_sec" {
entry:
  ret i32 %a
}

define dso_local signext i32 @foo3(i32 noundef signext %a) #0 {
entry:
  ret i32 %a
}

define dso_local signext i32 @foo4(i32 noundef signext %a) #0 {
entry:
  ret i32 %a
}
