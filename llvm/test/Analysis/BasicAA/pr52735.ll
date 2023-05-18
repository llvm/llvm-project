; RUN: opt %s -aa-pipeline=basic-aa -passes=aa-eval -print-all-alias-modref-info -disable-output 2>&1 | FileCheck %s
;
; Generated from:
;
; int foo() {
;   int v;
;   asm goto("movl $1, %0" : "=m"(v)::: out);
; out:
;   return v;
; }

target triple = "x86_64-unknown-linux-gnu"

; CHECK: Both ModRef:  Ptr: i32* %v	<->  callbr void asm "movl $$1, $0", "=*m,!i,~{dirflag},~{fpsr},~{flags}"(ptr nonnull elementtype(i32) %v)


define dso_local i32 @foo() {
entry:
  %v = alloca i32, align 4
  callbr void asm "movl $$1, $0", "=*m,!i,~{dirflag},~{fpsr},~{flags}"(ptr elementtype(i32) nonnull %v)
          to label %asm.fallthrough [label %out]

asm.fallthrough:
  br label %out

out:
  %0 = load i32, ptr %v, align 4
  ret i32 %0
}
