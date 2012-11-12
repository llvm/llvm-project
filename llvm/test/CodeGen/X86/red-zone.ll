; RUN: llc < %s -mcpu=generic -mtriple=x86_64-linux | FileCheck %s

; First without noredzone.
; CHECK: f0:
; CHECK: -4(%rsp)
; CHECK: -4(%rsp)
; CHECK: ret
define x86_fp80 @f0(float %f) nounwind readnone {
entry:
	%0 = fpext float %f to x86_fp80		; <x86_fp80> [#uses=1]
	ret x86_fp80 %0
}

; Then with noredzone.
; CHECK: f1:
; CHECK: subq $4, %rsp
; CHECK: (%rsp)
; CHECK: (%rsp)
; CHECK: addq $4, %rsp
; CHECK: ret
define x86_fp80 @f1(float %f) nounwind readnone noredzone {
entry:
	%0 = fpext float %f to x86_fp80		; <x86_fp80> [#uses=1]
	ret x86_fp80 %0
}
