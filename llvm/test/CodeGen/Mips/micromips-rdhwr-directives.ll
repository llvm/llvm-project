; RUN: llc -mtriple=mipsel -mcpu=mips32 -relocation-model=static < %s \
; RUN:   -mattr=+micromips | FileCheck %s

@a = external thread_local global i32

define i32 @foo() nounwind readonly {
entry:
; CHECK: .set  push
; CHECK: .set  mips32r2
; CHECK: rdhwr
; CHECK: .set  pop

  %0 = load i32, i32* @a, align 4
  ret i32 %0
}
