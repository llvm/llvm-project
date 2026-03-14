; RUN: llc < %s -mtriple=x86_64-scei-ps4 | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-sie-ps5  | FileCheck %s

declare i32 @personality(...)

; Check that after the (implicitly noreturn) unwind call, there is
; another instruction. It was easy to produce 'ud2' so we check for that.
define void @foo1() personality ptr @personality {
; CHECK-LABEL: foo1:
; CHECK: .cfi_startproc
; CHECK: callq bar
; CHECK: retq
; Check for 'ud2' between noreturn call and function end.
; CHECK: callq _Unwind_Resume
; CHECK-NEXT: ud2
; CHECK-NEXT: .Lfunc_end0:
    invoke void @bar()
        to label %normal
        unwind label %catch
normal:
    ret void
catch:
    %1 = landingpad { ptr, i32 } cleanup
    resume { ptr, i32 } %1
}

declare void @bar() #0

; Similar check after an explicit noreturn call.
define void @foo2() {
; CHECK-LABEL: foo2:
; CHECK: callq bar
; CHECK-NEXT: ud2
; CHECK-NEXT: .Lfunc_end1:
    tail call void @bar()
    unreachable
}

attributes #0 = { noreturn }
