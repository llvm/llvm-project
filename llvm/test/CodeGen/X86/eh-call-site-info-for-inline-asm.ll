; RUN: llc < %s -mtriple=x86_64-pc-linux | FileCheck %s
; Test that we emit call site info for inline asm calls that may unwind.

declare i32 @__my_personality_v0(...)
declare void @might_throw()

define void @foo() personality ptr @__my_personality_v0 {
; CHECK: .cfi_personality 3, __my_personality_v0
; CHECK:      .Lcst_begin0:
; CHECK-NEXT: .uleb128 .Lfunc_begin0-.Lfunc_begin0
; CHECK-NEXT: .uleb128 .Ltmp0-.Lfunc_begin0
; CHECK-NEXT: .byte   0
; CHECK-NEXT: .byte   0
; CHECK-NEXT: .uleb128 .Ltmp0-.Lfunc_begin0
; CHECK-NEXT: .uleb128 .Ltmp1-.Ltmp0
; CHECK-NEXT: .uleb128 .Ltmp2-.Lfunc_begin0
; CHECK-NEXT: .byte   0
; CHECK-NEXT: .uleb128 .Ltmp1-.Lfunc_begin0
; CHECK-NEXT: .uleb128 .Lfunc_end0-.Ltmp1
; CHECK-NEXT: .byte   0
; CHECK-NEXT: .byte   0
; CHECK-NEXT: .Lcst_end0:

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
