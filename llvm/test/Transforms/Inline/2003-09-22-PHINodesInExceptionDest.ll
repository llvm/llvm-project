; RUN: llvm-as < %s | opt -inline -disable-output

define i32 @main() {
entry:
        invoke void @__main( )
                        to label %Call2Invoke unwind label %LongJmpBlkPre

Call2Invoke:            ; preds = %entry
        br label %LongJmpBlkPre

LongJmpBlkPre:          ; preds = %Call2Invoke, %entry
        %i.3 = phi i32 [ 0, %entry ], [ 0, %Call2Invoke ]               ; <i32> [#uses=0]
        ret i32 0
}

define void @__main() {
        call void @__llvm_getGlobalCtors( )
        call void @__llvm_getGlobalDtors( )
        ret void
}

declare void @__llvm_getGlobalCtors()

declare void @__llvm_getGlobalDtors()

