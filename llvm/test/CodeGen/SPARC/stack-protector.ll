; RUN: llc -mtriple=sparc-unknown-linux < %s | FileCheck %s --check-prefix=LINUX-32
; RUN: llc -mtriple=sparc64-unknown-linux < %s | FileCheck %s --check-prefix=LINUX-64
; RUN: llc -mtriple=sparc-unknown-solaris < %s | FileCheck %s --check-prefix=GENERIC
; RUN: llc -mtriple=sparc64-unknown-solaris < %s | FileCheck %s --check-prefix=GENERIC

; LINUX-32: ld [%g7+20], [[REG1:%[ilo][0-9]*]]
; LINUX-64: ldx [%g7+40], [[REG1:%[ilo][0-9]*]]
; LINUX-32-NOT: __stack_chk_guard
; LINUX-64-NOT: __stack_chk_guard
; GENERIC: __stack_chk_guard

@"\01LC" = internal constant [11 x i8] c"buf == %s\0A\00"		; <[11 x i8]*> [#uses=1]

define void @test(ptr %a) nounwind ssp {
entry:
	%a_addr = alloca ptr		; <i8**> [#uses=2]
	%buf = alloca [8 x i8]		; <[8 x i8]*> [#uses=2]
  %"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	store ptr %a, ptr %a_addr
	%buf1 = bitcast ptr %buf to ptr		; <i8*> [#uses=1]
	%0 = load ptr, ptr %a_addr, align 4		; <i8*> [#uses=1]
	%1 = call ptr @strcpy(ptr %buf1, ptr %0) nounwind		; <i8*> [#uses=0]
  %buf2 = bitcast ptr %buf to ptr		; <i8*> [#uses=1]
	%2 = call i32 (ptr, ...) @printf(ptr @"\01LC", ptr %buf2) nounwind		; <i32> [#uses=0]
	br label %return

return:		; preds = %entry
	ret void
}

declare ptr @strcpy(ptr, ptr) nounwind

declare i32 @printf(ptr, ...) nounwind
