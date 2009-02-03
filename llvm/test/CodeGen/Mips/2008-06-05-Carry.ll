; RUN: llvm-as < %s | llc -march=mips -f -o %t
; RUN: grep subu %t | count 2
; RUN: grep addu %t | count 4

target datalayout =
"e-p:32:32:32-i1:8:8-i8:8:32-i16:16:32-i32:32:32-i64:32:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64"
target triple = "mipsallegrexel-psp-elf"

define i64 @add64(i64 %u, i64 %v) nounwind  {
entry:
	%tmp2 = add i64 %u, %v	
  ret i64 %tmp2
}

define i64 @sub64(i64 %u, i64 %v) nounwind  {
entry:
  %tmp2 = sub i64 %u, %v
  ret i64 %tmp2
}
