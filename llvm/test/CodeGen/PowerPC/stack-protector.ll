; RUN: llc -verify-machineinstrs -mtriple=powerpc-unknown-linux < %s | FileCheck -check-prefix=LINUX32 %s
; RUN: llc -verify-machineinstrs -mtriple=powerpc64-unknown-linux < %s | FileCheck -check-prefix=LINUX64 %s
; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-unknown-linux < %s | FileCheck -check-prefix=LINUX64 %s
; RUN: llc -verify-machineinstrs -mtriple=powerpc-ibm-aix-xcoff < %s | FileCheck -check-prefix=AIX32 %s
; RUN: llc -verify-machineinstrs -mtriple=powerpc64-ibm-aix-xcoff < %s | FileCheck -check-prefix=AIX64 %s
; RUN: llc -verify-machineinstrs -mtriple=powerpc-unknown-freebsd13 < %s | FileCheck -check-prefix=FREEBSD32 %s
; RUN: llc -verify-machineinstrs -mtriple=powerpcle-unknown-freebsd13 < %s | FileCheck -check-prefix=FREEBSD32 %s
; RUN: llc -verify-machineinstrs -mtriple=powerpc64-unknown-freebsd13 < %s | FileCheck -check-prefix=FREEBSD64 %s
; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-unknown-freebsd13 < %s | FileCheck -check-prefix=FREEBSD64 %s

; LINUX32: lwz [[#]], -28680(2)
; LINUX64: ld [[#]], -28688(13)
; AIX32: lwz {{.*}}__ssp_canary_word
; AIX64: ld {{.*}}__ssp_canary_word
; FREEBSD32: lwz [[#]], __stack_chk_guard@l([[#]])
; FREEBSD64: ld [[#]], .LC0@toc@l([[#]])

; LINUX32: __stack_chk_fail
; LINUX64: __stack_chk_fail
; AIX32: __stack_chk_fail
; AIX64: __stack_chk_fail
; FREEBSD32: bl __stack_chk_fail
; FREEBSD64: bl __stack_chk_fail

@"\01LC" = internal constant [11 x i8] c"buf == %s\0A\00"		; <ptr> [#uses=1]

define void @test(ptr %a) nounwind ssp {
entry:
	%a_addr = alloca ptr		; <ptr> [#uses=2]
	%buf = alloca [8 x i8]		; <ptr> [#uses=2]
  %"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	store ptr %a, ptr %a_addr
	%0 = load ptr, ptr %a_addr, align 4		; <ptr> [#uses=1]
	%1 = call ptr @strcpy(ptr %buf, ptr %0) nounwind		; <ptr> [#uses=0]
	%2 = call i32 (ptr, ...) @printf(ptr @"\01LC", ptr %buf) nounwind		; <i32> [#uses=0]
	br label %return

return:		; preds = %entry
	ret void
}

declare ptr @strcpy(ptr, ptr) nounwind

declare i32 @printf(ptr, ...) nounwind
