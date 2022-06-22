; RUN: llc < %s -mtriple=x86_64-apple-darwin | FileCheck %s
; rdar://6647639

	%struct.FetchPlanHeader = type { ptr, ptr, i32, ptr, ptr, ptr, ptr, ptr, ptr, %struct.__attributeDescriptionFlags }
	%struct.NSArray = type { %struct.NSObject }
	%struct.NSAutoreleasePool = type { %struct.NSObject, ptr, ptr, ptr, ptr }
	%struct.NSObject = type { ptr }
	%struct.__attributeDescriptionFlags = type <{ i32 }>
	%struct._message_ref_t = type { ptr, ptr }
	%struct.objc_selector = type opaque
@"\01l_objc_msgSend_fixup_alloc" = external global %struct._message_ref_t, align 16		; <ptr> [#uses=2]

define ptr @newFetchedRowsForFetchPlan_MT(ptr %fetchPlan, ptr %selectionMethod, ptr %selectionParameter) ssp personality ptr @__gxx_personality_v0 {
; CHECK-LABEL: newFetchedRowsForFetchPlan_MT:
; CHECK:       ## %bb.0: ## %entry
; CHECK-NEXT:    pushq %rax
; CHECK-NEXT:    .cfi_def_cfa_offset 16
; CHECK-NEXT:  Ltmp0:
; CHECK-NEXT:    movq l_objc_msgSend_fixup_alloc@{{.*}}(%rip), %rsi
; CHECK-NEXT:    xorl %eax, %eax
; CHECK-NEXT:    xorl %edi, %edi
; CHECK-NEXT:    callq *%rax
; CHECK-NEXT:  Ltmp1:
; CHECK-NEXT:  ## %bb.1: ## %invcont
; CHECK-NEXT:  Ltmp2:
; CHECK-NEXT:    movq %rax, %rdi
; CHECK-NEXT:    xorl %esi, %esi
; CHECK-NEXT:    xorl %eax, %eax
; CHECK-NEXT:    callq _objc_msgSend
; CHECK-NEXT:  Ltmp3:
; CHECK-NEXT:  ## %bb.2: ## %invcont26
; CHECK-NEXT:  Ltmp4:
; CHECK-NEXT:    movq l_objc_msgSend_fixup_alloc@{{.*}}(%rip), %rsi
; CHECK-NEXT:    xorl %eax, %eax
; CHECK-NEXT:    xorl %edi, %edi
; CHECK-NEXT:    callq *%rax
; CHECK-NEXT:  Ltmp5:
; CHECK-NEXT:  ## %bb.3: ## %invcont27
; CHECK-NEXT:    ud2
; CHECK-NEXT:  LBB0_4: ## %lpad
; CHECK-NEXT:  Ltmp6:
; CHECK-NEXT:    ud2
; CHECK-NEXT:  Lfunc_end0:
entry:
	%0 = invoke ptr null(ptr null, ptr @"\01l_objc_msgSend_fixup_alloc")
			to label %invcont unwind label %lpad		; <ptr> [#uses=1]

invcont:		; preds = %entry
	%1 = invoke ptr (ptr, ptr, ...) @objc_msgSend(ptr %0, ptr null)
			to label %invcont26 unwind label %lpad		; <ptr> [#uses=0]

invcont26:		; preds = %invcont
	%2 = invoke ptr null(ptr null, ptr @"\01l_objc_msgSend_fixup_alloc")
			to label %invcont27 unwind label %lpad		; <ptr> [#uses=0]

invcont27:		; preds = %invcont26
	unreachable

lpad:		; preds = %invcont26, %invcont, %entry
	%pool.1 = phi ptr [ null, %entry ], [ null, %invcont ], [ null, %invcont26 ]		; <ptr> [#uses=0]
        %exn = landingpad {ptr, i32}
                 cleanup
	unreachable
}

declare ptr @objc_msgSend(ptr, ptr, ...)

declare i32 @__gxx_personality_v0(...)
