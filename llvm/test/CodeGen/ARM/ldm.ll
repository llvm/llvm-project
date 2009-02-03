; RUN: llvm-as < %s | llc -march=arm | \
; RUN:   grep ldmia | count 2
; RUN: llvm-as < %s | llc -march=arm | \
; RUN:   grep ldmib | count 1
; RUN: llvm-as < %s | llc -mtriple=arm-apple-darwin | \
; RUN:   grep {ldmfd sp\!} | count 3

@X = external global [0 x i32]          ; <[0 x i32]*> [#uses=5]

define i32 @t1() {
        %tmp = load i32* getelementptr ([0 x i32]* @X, i32 0, i32 0)            ; <i32> [#uses=1]
        %tmp3 = load i32* getelementptr ([0 x i32]* @X, i32 0, i32 1)           ; <i32> [#uses=1]
        %tmp4 = tail call i32 @f1( i32 %tmp, i32 %tmp3 )                ; <i32> [#uses=1]
        ret i32 %tmp4
}

define i32 @t2() {
        %tmp = load i32* getelementptr ([0 x i32]* @X, i32 0, i32 2)            ; <i32> [#uses=1]
        %tmp3 = load i32* getelementptr ([0 x i32]* @X, i32 0, i32 3)           ; <i32> [#uses=1]
        %tmp5 = load i32* getelementptr ([0 x i32]* @X, i32 0, i32 4)           ; <i32> [#uses=1]
        %tmp6 = tail call i32 @f2( i32 %tmp, i32 %tmp3, i32 %tmp5 )             ; <i32> [#uses=1]
        ret i32 %tmp6
}

define i32 @t3() {
        %tmp = load i32* getelementptr ([0 x i32]* @X, i32 0, i32 1)            ; <i32> [#uses=1]
        %tmp3 = load i32* getelementptr ([0 x i32]* @X, i32 0, i32 2)           ; <i32> [#uses=1]
        %tmp5 = load i32* getelementptr ([0 x i32]* @X, i32 0, i32 3)           ; <i32> [#uses=1]
        %tmp6 = tail call i32 @f2( i32 %tmp, i32 %tmp3, i32 %tmp5 )             ; <i32> [#uses=1]
        ret i32 %tmp6
}

declare i32 @f1(i32, i32)

declare i32 @f2(i32, i32, i32)
