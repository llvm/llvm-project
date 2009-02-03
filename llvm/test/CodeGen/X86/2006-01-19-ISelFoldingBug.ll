; RUN: llvm-as < %s | llc -march=x86 | \
; RUN:   grep shld | count 1
;
; Check that the isel does not fold the shld, which already folds a load
; and has two uses, into a store.

@A = external global i32                ; <i32*> [#uses=2]

define i32 @test5(i32 %B, i8 %C) {
        %tmp.1 = load i32* @A           ; <i32> [#uses=1]
        %shift.upgrd.1 = zext i8 %C to i32              ; <i32> [#uses=1]
        %tmp.2 = shl i32 %tmp.1, %shift.upgrd.1         ; <i32> [#uses=1]
        %tmp.3 = sub i8 32, %C          ; <i8> [#uses=1]
        %shift.upgrd.2 = zext i8 %tmp.3 to i32          ; <i32> [#uses=1]
        %tmp.4 = lshr i32 %B, %shift.upgrd.2            ; <i32> [#uses=1]
        %tmp.5 = or i32 %tmp.4, %tmp.2          ; <i32> [#uses=2]
        store i32 %tmp.5, i32* @A
        ret i32 %tmp.5
}

