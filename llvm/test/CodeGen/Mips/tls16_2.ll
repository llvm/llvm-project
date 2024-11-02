; RUN: llc  -mtriple=mipsel-- -mattr=mips16 -relocation-model=pic -O3 < %s | FileCheck %s -check-prefix=PIC16

@f.i = internal thread_local unnamed_addr global i32 1, align 4

define ptr @f(ptr nocapture %a) nounwind {
entry:
  %0 = load i32, ptr @f.i, align 4
  %inc = add nsw i32 %0, 1
  store i32 %inc, ptr @f.i, align 4
  %1 = inttoptr i32 %inc to ptr
; PIC16: addiu	${{[0-9]+}}, %tlsldm(f.i)
  ret ptr %1
}


