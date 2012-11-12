; RUN: opt < %s -inline -disable-output

define i32 @main() {
entry:
        invoke void @__main( )
                        to label %Call2Invoke unwind label %LongJmpBlkPre

Call2Invoke:            ; preds = %entry
        br label %exit

LongJmpBlkPre:          ; preds = %Call2Invoke, %entry
        %i.3 = phi i32 [ 0, %entry ], [ 0, %Call2Invoke ]               ; <i32> [#uses=0]
        %exn = landingpad {i8*, i32} personality i32 (...)* @__gxx_personality_v0
                 cleanup
        br label %exit

exit:
        ret i32 0
}

define void @__main() {
        call void @__llvm_getGlobalCtors( )
        call void @__llvm_getGlobalDtors( )
        ret void
}

declare i32 @__gxx_personality_v0(...)

declare void @__llvm_getGlobalCtors()

declare void @__llvm_getGlobalDtors()

