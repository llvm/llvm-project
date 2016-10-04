; RUN: llc  -mtriple=mipsel-- -mattr=mips16 -relocation-model=pic -O3 < %s | FileCheck %s -check-prefix=PIC16

@f.i = internal thread_local unnamed_addr global i32 1, align 4

define i8* @f(i8* nocapture %a) nounwind {
entry:
  %0 = load i32, i32* @f.i, align 4
  %inc = add nsw i32 %0, 1
  store i32 %inc, i32* @f.i, align 4
  %1 = inttoptr i32 %inc to i8*
; PIC16: addiu	${{[0-9]+}}, %tlsldm(f.i)
  ret i8* %1
}


