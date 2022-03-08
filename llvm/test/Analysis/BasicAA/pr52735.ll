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

; CHECK: MayAlias: i32* %v, void (i32*, i8*)* asm "movl $$1, $0", "=*m,X,~{dirflag},~{fpsr},~{flags}"

define dso_local i32 @foo() {
entry:
  %v = alloca i32, align 4
  %0 = bitcast i32* %v to i8*
  callbr void asm "movl $$1, $0", "=*m,X,~{dirflag},~{fpsr},~{flags}"(i32* elementtype(i32) nonnull %v, i8* blockaddress(@foo, %out))
          to label %asm.fallthrough [label %out]

asm.fallthrough:
  br label %out

out:
  %1 = load i32, i32* %v, align 4
  ret i32 %1
}
