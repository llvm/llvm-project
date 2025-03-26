; RUN: llc -mtriple=mips -mattr=single-float  < %s | FileCheck %s

define double @int2fp(i32 %a) nounwind {
entry:
; CHECK: int2fp
; CHECK: __floatsidf
	sitofp i32 %a to double		; <double>:0 [#uses=1]
	ret double %0
}

define double @uint2double(i32 %a) nounwind {
entry:
; CHECK: uint2double
; CHECK: __floatunsidf
	uitofp i32 %a to double		; <double>:0 [#uses=1]
	ret double %0
}

define i32 @double2int(double %a) nounwind {
entry:
; CHECK: double2int
; CHECK: __fixdfsi
  fptosi double %a to i32   ; <i32>:0 [#uses=1]
  ret i32 %0
}

define i32 @double2uint(double %a) nounwind {
entry:
; CHECK: double2uint
; CHECK: __fixunsdfsi
  fptoui double %a to i32   ; <i32>:0 [#uses=1]
  ret i32 %0
}

