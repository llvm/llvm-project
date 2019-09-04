; RUN: llc < %s -mtriple=x86_64-pc-mingw32 | FileCheck %s -check-prefix=WIN64

declare void @bar()
declare void @baz()
declare i32 @personality(...)

; Check for 'nop' between the last call and the epilogue.
define void @foo1() personality i32 (...)* @personality {

    invoke void @bar()
        to label %normal
        unwind label %catch

normal:
    ret void

catch:
    %1 = landingpad { i8*, i32 } cleanup
    resume { i8*, i32 } %1
}
; WIN64-LABEL: foo1:
; WIN64: .seh_proc foo1
; WIN64: callq bar
; WIN64: nop
; WIN64: addq ${{[0-9]+}}, %rsp
; WIN64: retq
; Check for 'ud2' after noreturn call
; WIN64: callq _Unwind_Resume
; WIN64-NEXT: ud2
; WIN64: .seh_endproc


; Check it still works when blocks are reordered.
@something = global i32 0
define void @foo2(i1 zeroext %cond ) {
    br i1 %cond, label %a, label %b, !prof !0
a:
    call void @bar()
    br label %done
b:
    call void @baz()
    store i32 0, i32* @something
    br label %done
done:
    ret void
}
!0 = !{!"branch_weights", i32 100, i32 0}
; WIN64-LABEL: foo2:
; WIN64: callq bar
; WIN64: nop
; WIN64: addq ${{[0-9]+}}, %rsp
; WIN64: retq


; Check nop is not emitted when call is not adjacent to epilogue.
define i32 @foo3() {
    call void @bar()
    ret i32 0
}
; WIN64-LABEL: foo3:
; WIN64: callq bar
; WIN64: xorl
; WIN64-NOT: nop
; WIN64: addq ${{[0-9]+}}, %rsp
; WIN64: retq
