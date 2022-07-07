; RUN: opt -passes=pgo-instr-gen -S 2>&1 < %s | FileCheck %s

define i32 @a() {
entry:
; CHECK-NOT: ptrtoint void (i8*)* asm sideeffect
; CHECK: callbr void asm sideeffect
  %retval = alloca i32, align 4
  callbr void asm sideeffect "", "!i,~{dirflag},~{fpsr},~{flags}"() #1
          to label %asm.fallthrough [label %b]

asm.fallthrough:
  br label %b

b:
  %0 = load i32, i32* %retval, align 4
  ret i32 %0
}
