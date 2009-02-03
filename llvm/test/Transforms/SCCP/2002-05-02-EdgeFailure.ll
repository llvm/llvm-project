; edgefailure - This function illustrates how SCCP is not doing it's job.  This
; function should be optimized almost completely away: the loop should be
; analyzed to detect that the body executes exactly once, and thus the branch
; can be eliminated and code becomes trivially dead.  This is distilled from a
; real benchmark (mst from Olden benchmark, MakeGraph function).  When SCCP is
; fixed, this should be eliminated by a single SCCP application.
;
; RUN: llvm-as < %s | opt -sccp | llvm-dis | not grep loop

define i32* @test() {
bb1:
	%A = malloc i32		; <i32*> [#uses=2]
	br label %bb2
bb2:		; preds = %bb2, %bb1
        ;; Always 0
	%i = phi i32 [ %i2, %bb2 ], [ 0, %bb1 ]		; <i32> [#uses=2]
        ;; Always 1
	%i2 = add i32 %i, 1		; <i32> [#uses=2]
	store i32 %i, i32* %A
        ;; Always false
  	%loop = icmp sle i32 %i2, 0		; <i1> [#uses=1]
	br i1 %loop, label %bb2, label %bb3
bb3:		; preds = %bb2
	ret i32* %A
}

