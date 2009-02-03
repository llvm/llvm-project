; RUN: llvm-as < %s | llc -march=mips -f -o %t
; RUN: grep mfhi  %t | count 1
; RUN: grep mflo  %t | count 1
; RUN: grep multu %t | count 1

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:32-i16:16:32-i32:32:32-i64:32:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64"
target triple = "mipsallegrexel-psp-elf"
	%struct.DWstruct = type { i32, i32 }

define i32 @A0(i32 %u, i32 %v) nounwind  {
entry:
	%asmtmp = tail call %struct.DWstruct asm "multu $2,$3", "={lo},={hi},d,d"( i32 %u, i32 %v ) nounwind
	%asmresult = extractvalue %struct.DWstruct %asmtmp, 0
	%asmresult1 = extractvalue %struct.DWstruct %asmtmp, 1		; <i32> [#uses=1]
  %res = add i32 %asmresult, %asmresult1
	ret i32 %res
}
