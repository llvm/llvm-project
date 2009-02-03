; RUN: llvm-as < %s | llc -march=xcore > %t1.s
;; This caused a compilation failure since the
;; address arithmetic was folded into the LDWSP instruction,
;; resulting in a negative offset which eliminateFrameIndex was
;; unable to eliminate.
define i32 @test(i32 %bar) nounwind readnone {
entry:
        %bar_addr = alloca i32
        %0 = getelementptr i32* %bar_addr, i32 -1
        %1 = load i32* %0, align 4
        ret i32 %1
}
