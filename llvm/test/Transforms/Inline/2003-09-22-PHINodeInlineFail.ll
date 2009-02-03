; RUN: llvm-as < %s | opt -inline -disable-output

define i32 @main() {
entry:
        invoke void @__main( )
                        to label %LongJmpBlkPre unwind label %LongJmpBlkPre

LongJmpBlkPre:          ; preds = %entry, %entry
        %i.3 = phi i32 [ 0, %entry ], [ 0, %entry ]             ; <i32> [#uses=0]
        ret i32 0
}

define void @__main() {
        ret void
}

