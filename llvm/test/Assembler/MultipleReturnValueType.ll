; RUN: llvm-as < %s
; RUN: verify-uselistorder %s

        %struct.S_102 = type { float, float }

declare %struct.S_102 @f_102() nounwind

@callthis = external global ptr            ; <ptr> [#uses=50]


define void @foo() {
        store ptr @f_102, ptr @callthis, align 8
        ret void
}
