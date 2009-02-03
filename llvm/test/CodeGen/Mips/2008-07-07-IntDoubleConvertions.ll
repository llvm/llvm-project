; RUN: llvm-as < %s | llc -march=mips -f -o %t
; RUN: grep __floatsidf   %t | count 1
; RUN: grep __floatunsidf %t | count 1
; RUN: grep __fixdfsi %t | count 1
; RUN: grep __fixunsdfsi %t  | count 1

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:32-i16:16:32-i32:32:32-i64:32:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64"
target triple = "mipsallegrexel-psp-elf"

define double @int2fp(i32 %a) nounwind {
entry:
	sitofp i32 %a to double		; <double>:0 [#uses=1]
	ret double %0
}

define double @uint2double(i32 %a) nounwind {
entry:
	uitofp i32 %a to double		; <double>:0 [#uses=1]
	ret double %0
}

define i32 @double2int(double %a) nounwind {
entry:
  fptosi double %a to i32   ; <i32>:0 [#uses=1]
  ret i32 %0
}

define i32 @double2uint(double %a) nounwind {
entry:
  fptoui double %a to i32   ; <i32>:0 [#uses=1]
  ret i32 %0
}

