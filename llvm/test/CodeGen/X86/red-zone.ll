; RUN: llc < %s -mcpu=generic -mtriple=x86_64-linux | FileCheck %s

@flags_gv = global i64 0

; First without noredzone.
; CHECK-LABEL: f0:
; CHECK: -4(%rsp)
; CHECK: -4(%rsp)
; CHECK: ret
define x86_fp80 @f0(float %f) nounwind {
entry:
	%0 = fpext float %f to x86_fp80		; <x86_fp80> [#uses=1]
	ret x86_fp80 %0
}

; Then with noredzone.
; CHECK-LABEL: f1:
; CHECK: subq $4, %rsp
; CHECK: (%rsp)
; CHECK: (%rsp)
; CHECK: addq $4, %rsp
; CHECK: ret
define x86_fp80 @f1(float %f) nounwind noredzone {
entry:
	%0 = fpext float %f to x86_fp80		; <x86_fp80> [#uses=1]
	ret x86_fp80 %0
}

declare i64 @llvm.x86.flags.read.u64()
declare void @llvm.x86.flags.write.u64(i64)


; pushfq and popfq prevent redzones.
; CHECK-LABEL: norz_flags_read:
; CHECK: subq ${{[0-9]+}}, %rsp
; CHECK: pushfq
; CHECK: popq
; CHECK: (%rsp)
; CHECK: (%rsp)
; CHECK: ret
define x86_fp80 @norz_flags_read(float %f) nounwind {
entry:
  %flags = call i64 @llvm.x86.flags.read.u64()
  store i64 %flags, ptr @flags_gv
  %0 = fpext float %f to x86_fp80
  ret x86_fp80 %0
}

; CHECK-LABEL: norz_flags_write:
; CHECK: subq ${{[0-9]+}}, %rsp
; CHECK: pushq
; CHECK: popfq
; CHECK: (%rsp)
; CHECK: (%rsp)
; CHECK: ret
define x86_fp80 @norz_flags_write(float %f, i64 %flags) nounwind {
entry:
  call void @llvm.x86.flags.write.u64(i64 %flags)
  %0 = fpext float %f to x86_fp80
  ret x86_fp80 %0
}
