; RUN: llvm-as < %s | llc -march=c | grep {\\* *volatile *\\*}

@G = external global void ()*           ; <void ()**> [#uses=2]

define void @test() {
        volatile store void ()* @test, void ()** @G
        volatile load void ()** @G              ; <void ()*>:1 [#uses=0]
        ret void
}

