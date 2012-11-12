; RUN: llc < %s -march=x86-64 | grep lea | count 12

; This testcase was written to demonstrate an instruction-selection problem,
; however it also happens to expose a limitation in the DAGCombiner's
; expression reassociation which causes it to miss opportunities for
; constant folding due to the intermediate adds having multiple uses.
; The Reassociate pass has similar limitations. If these limitations are
; fixed, the test commands above will need to be updated to expect fewer
; lea instructions.

@g0 = weak global [1000 x i32] zeroinitializer, align 32		; <[1000 x i32]*> [#uses=8]
@g1 = weak global [1000 x i32] zeroinitializer, align 32		; <[1000 x i32]*> [#uses=7]

define void @foo() {
entry:
	%tmp4 = load i32* getelementptr ([1000 x i32]* @g0, i32 0, i32 0)		; <i32> [#uses=1]
	%tmp8 = load i32* getelementptr ([1000 x i32]* @g1, i32 0, i32 0)		; <i32> [#uses=1]
	%tmp9 = add i32 %tmp4, 1		; <i32> [#uses=1]
	%tmp10 = add i32 %tmp9, %tmp8		; <i32> [#uses=2]
	store i32 %tmp10, i32* getelementptr ([1000 x i32]* @g0, i32 0, i32 1)
	%tmp8.1 = load i32* getelementptr ([1000 x i32]* @g1, i32 0, i32 1)		; <i32> [#uses=1]
	%tmp9.1 = add i32 %tmp10, 1		; <i32> [#uses=1]
	%tmp10.1 = add i32 %tmp9.1, %tmp8.1		; <i32> [#uses=2]
	store i32 %tmp10.1, i32* getelementptr ([1000 x i32]* @g0, i32 0, i32 2)
	%tmp8.2 = load i32* getelementptr ([1000 x i32]* @g1, i32 0, i32 2)		; <i32> [#uses=1]
	%tmp9.2 = add i32 %tmp10.1, 1		; <i32> [#uses=1]
	%tmp10.2 = add i32 %tmp9.2, %tmp8.2		; <i32> [#uses=2]
	store i32 %tmp10.2, i32* getelementptr ([1000 x i32]* @g0, i32 0, i32 3)
	%tmp8.3 = load i32* getelementptr ([1000 x i32]* @g1, i32 0, i32 3)		; <i32> [#uses=1]
	%tmp9.3 = add i32 %tmp10.2, 1		; <i32> [#uses=1]
	%tmp10.3 = add i32 %tmp9.3, %tmp8.3		; <i32> [#uses=2]
	store i32 %tmp10.3, i32* getelementptr ([1000 x i32]* @g0, i32 0, i32 4)
	%tmp8.4 = load i32* getelementptr ([1000 x i32]* @g1, i32 0, i32 4)		; <i32> [#uses=1]
	%tmp9.4 = add i32 %tmp10.3, 1		; <i32> [#uses=1]
	%tmp10.4 = add i32 %tmp9.4, %tmp8.4		; <i32> [#uses=2]
	store i32 %tmp10.4, i32* getelementptr ([1000 x i32]* @g0, i32 0, i32 5)
	%tmp8.5 = load i32* getelementptr ([1000 x i32]* @g1, i32 0, i32 5)		; <i32> [#uses=1]
	%tmp9.5 = add i32 %tmp10.4, 1		; <i32> [#uses=1]
	%tmp10.5 = add i32 %tmp9.5, %tmp8.5		; <i32> [#uses=2]
	store i32 %tmp10.5, i32* getelementptr ([1000 x i32]* @g0, i32 0, i32 6)
	%tmp8.6 = load i32* getelementptr ([1000 x i32]* @g1, i32 0, i32 6)		; <i32> [#uses=1]
	%tmp9.6 = add i32 %tmp10.5, 1		; <i32> [#uses=1]
	%tmp10.6 = add i32 %tmp9.6, %tmp8.6		; <i32> [#uses=1]
	store i32 %tmp10.6, i32* getelementptr ([1000 x i32]* @g0, i32 0, i32 7)
	ret void
}

