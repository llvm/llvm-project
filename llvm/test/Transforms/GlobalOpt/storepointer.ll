; RUN: llvm-as < %s | opt -globalopt | llvm-dis | not grep global

@G = internal global void ()* null              ; <void ()**> [#uses=2]

define internal void @Actual() {
        ret void
}

define void @init() {
        store void ()* @Actual, void ()** @G
        ret void
}

define void @doit() {
        %FP = load void ()** @G         ; <void ()*> [#uses=1]
        call void %FP( )
        ret void
}

