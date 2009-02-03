; RUN: llvm-as < %s | llvm-dis > %t1.ll
; RUN: llvm-as %t1.ll -o - | llvm-dis > %t2.ll
; RUN: diff %t1.ll %t2.ll

        %int = type i32

define i32 @squared(i32 %i0) {
        switch i32 %i0, label %Default [
                 i32 1, label %Case1
                 i32 2, label %Case2
                 i32 4, label %Case4
        ]

Default:                ; preds = %0
        ret i32 -1

Case1:          ; preds = %0
        ret i32 1

Case2:          ; preds = %0
        ret i32 4

Case4:          ; preds = %0
        ret i32 16
}

