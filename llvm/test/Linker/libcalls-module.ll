; RUN: llvm-link %s %S/Inputs/has-libcalls.ll -S -o - 2>%t.a.err | FileCheck %s
; RUN: llvm-link %S/Inputs/has-libcalls.ll %s -S -o - 2>%t.a.err | FileCheck %s

target triple = "x86_64-unknown-linux-gnu"

; CHECK: define void @foo() #[[ATTR0:[0-9]+]]
define void @foo() #0 {
  ret void
}

attributes #0 = { noinline }

; CHECK: attributes #[[ATTR0]] = { nobuiltin noinline "no-builtins" }

; CHECK: !llvm.module.flags = !{!0}
; CHECK: !0 = !{i32 4, !"llvm-libcalls", i32 1}
