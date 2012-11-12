; RUN: llc < %s -march=mips -mcpu=mips32r2 | FileCheck %s 
; RUN: llc < %s -march=mips64 -mcpu=mips64r2 | FileCheck %s 

define signext i8 @A(i8 %e.0, i8 signext %sum)  nounwind {
entry:
; CHECK: seb
	add i8 %sum, %e.0		; <i8>:0 [#uses=1]
	ret i8 %0
}

define signext i16 @B(i16 %e.0, i16 signext %sum) nounwind {
entry:
; CHECK: seh
	add i16 %sum, %e.0		; <i16>:0 [#uses=1]
	ret i16 %0
}

