; RUN: llvm-as < %s | opt -dse | llvm-dis | \
; RUN:    not grep {store i8}
; Ensure that the dead store is deleted in this case.  It is wholely
; overwritten by the second store.
define i32 @test() {
        %V = alloca i32         ; <i32*> [#uses=3]
        %V2 = bitcast i32* %V to i8*            ; <i8*> [#uses=1]
        store i8 0, i8* %V2
        store i32 1234567, i32* %V
        %X = load i32* %V               ; <i32> [#uses=1]
        ret i32 %X
}

