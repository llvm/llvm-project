; RUN: llc < %s -mtriple=x86_64-pc-linux | FileCheck %s
; Test that we emit call site info for inline asm calls that may unwind.

declare i32 @__my_personality_v0(...)
declare void @might_throw()

define void @foo() personality ptr @__my_personality_v0 {
; CHECK-LABEL: foo:
; CHECK: [[FUNC_BEGIN:\.Lfunc_begin[0-9]+]]:
; CHECK: .cfi_personality 3, __my_personality_v0
; CHECK: callq might_throw
; CHECK: [[INVOKE_BEGIN:\.Ltmp[0-9]+]]: # EH_LABEL
; CHECK: callq might_throw
; CHECK: [[INVOKE_END:\.Ltmp[0-9]+]]: # EH_LABEL
; CHECK: [[LPAD:\.Ltmp[0-9]+]]: # EH_LABEL
; CHECK: callq _Unwind_Resume
; CHECK: [[FUNC_END:\.Lfunc_end[0-9]+]]:
; CHECK: .uleb128 [[CST_END:\.Lcst_end[0-9]+]]-[[CST_BEGIN:\.Lcst_begin[0-9]+]]
; CHECK-NEXT: [[CST_BEGIN]]:
; CHECK-NEXT: .uleb128 [[FUNC_BEGIN]]-[[FUNC_BEGIN]]
; CHECK-NEXT: .uleb128 [[INVOKE_BEGIN]]-[[FUNC_BEGIN]]
; CHECK-NEXT: .byte   0
; CHECK-NEXT: .byte   0
; CHECK-NEXT: .uleb128 [[INVOKE_BEGIN]]-[[FUNC_BEGIN]]
; CHECK-NEXT: .uleb128 [[INVOKE_END]]-[[INVOKE_BEGIN]]
; CHECK-NEXT: .uleb128 [[LPAD]]-[[FUNC_BEGIN]]
; CHECK-NEXT: .byte   0
; CHECK-NEXT: .uleb128 [[INVOKE_END]]-[[FUNC_BEGIN]]
; CHECK-NEXT: .uleb128 [[FUNC_END]]-[[INVOKE_END]]
; CHECK-NEXT: .byte   0
; CHECK-NEXT: .byte   0
; CHECK-NEXT: [[CST_END]]:

    ; An inline asm call that may unwind but has no landing pad.
    call void asm sideeffect alignstack inteldialect unwind "call ${0:P}", "X"(ptr @might_throw)

    ; An inline asm invoke with a landing pad.
    invoke void asm sideeffect alignstack inteldialect unwind
        "call ${0:P}", "X"(ptr @might_throw)
        to label %cont unwind label %lpad

cont:
    ret void

lpad:
    %eh = landingpad { ptr, i32 }
            cleanup
    resume { ptr, i32 } %eh
}
