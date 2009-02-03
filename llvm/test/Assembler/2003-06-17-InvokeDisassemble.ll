; RUN: llvm-as < %s | llvm-dis

define void @test() {
        invoke void @test( )
                        to label %Next unwind label %Next

Next:           ; preds = %0, %0
        ret void
}
