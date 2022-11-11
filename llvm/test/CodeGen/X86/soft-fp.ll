; RUN: llc < %s -mtriple=i686--    -mattr=+mmx,+sse,+soft-float \
; RUN:     | FileCheck %s --check-prefix=SOFT1 --check-prefix=CHECK
; RUN: llc < %s -mtriple=x86_64-- -mattr=+mmx,+sse2,+soft-float \
; RUN:     | FileCheck %s --check-prefix=SOFT2 --check-prefix=CHECK
; RUN: llc < %s -mtriple=x86_64-- -mattr=+mmx,+sse \
; RUN:     | FileCheck %s --check-prefix=SSE1 --check-prefix=CHECK
; RUN: llc < %s -mtriple=x86_64-- -mattr=+mmx,+sse2 \
; RUN:     | FileCheck %s --check-prefix=SSE2 --check-prefix=CHECK
; RUN: llc < %s -mtriple=x86_64-gnux32 -mattr=+mmx,+sse2,+soft-float | FileCheck %s

; CHECK-NOT: xmm{{[0-9]+}}

%struct.__va_list_tag = type { i32, i32, ptr, ptr }

define i32 @t1(i32 %a, ...) nounwind {
entry:
	%va = alloca [1 x %struct.__va_list_tag], align 8		; <ptr> [#uses=2]
	call void @llvm.va_start(ptr %va)
	%va3 = getelementptr [1 x %struct.__va_list_tag], ptr %va, i64 0, i64 0		; <ptr> [#uses=1]
	call void @bar(ptr %va3) nounwind
	call void @llvm.va_end(ptr %va)
	ret i32 undef
; CHECK-LABEL: t1:
; CHECK:       ret{{[lq]}}
}

declare void @llvm.va_start(ptr) nounwind

declare void @bar(ptr)

declare void @llvm.va_end(ptr) nounwind

define float @t2(float %a, float %b) nounwind readnone {
entry:
	%0 = fadd float %a, %b		; <float> [#uses=1]
	ret float %0
; CHECK-LABEL: t2:
; SOFT1-NOT:   xmm{{[0-9]+}}
; SOFT2-NOT:   xmm{{[0-9]+}}
; SSE1:        xmm{{[0-9]+}}
; SSE2:        xmm{{[0-9]+}}
; CHECK:       ret{{[lq]}}
}

; soft-float means no SSE instruction and passing fp128 as pair of i64.
define fp128 @t3(fp128 %a, fp128 %b) nounwind readnone {
entry:
	%0 = fadd fp128 %b, %a
	ret fp128 %0
; CHECK-LABEL: t3:
; SOFT1-NOT:   xmm{{[0-9]+}}
; SOFT2-NOT:   xmm{{[0-9]+}}
; SSE1:        xmm{{[0-9]+}}
; SSE2:        xmm{{[0-9]+}}
; SOFT1:       ret{{[lq]}}
; SOFT2:       ret{{[lq]}}
; SSE1:       jmp __addtf3
; SSE2:       jmp __addtf3
}
