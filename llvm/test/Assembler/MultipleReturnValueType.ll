; RUN: llvm-as < %s

        %struct.S_102 = type { float, float }

declare %struct.S_102 @f_102() nounwind

@callthis = external global void ()*            ; <void ()**> [#uses=50]


define void @foo() {
        store void ()* bitcast (%struct.S_102 ()* @f_102 to void ()*), void ()** @callthis, align 8
        ret void
}
