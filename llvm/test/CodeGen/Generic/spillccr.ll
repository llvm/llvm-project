; RUN: llvm-as %s -o - | llc

; July 6, 2002 -- LLC Regression test
; This test case checks if the integer CC register %xcc (or %ccr)
; is correctly spilled.  The code fragment came from function
; MakeGraph in Olden-mst.
; The original code made all comparisons with 0, so that the %xcc
; register is not needed for the branch in the first basic block.
; Replace 0 with 1 in the first comparson so that the
; branch-on-register instruction cannot be used directly, i.e.,
; the %xcc register is needed for the first branch.
;

        %Graph = type %struct.graph_st*
        %Hash = type %struct.hash*
        %HashEntry = type %struct.hash_entry*
        %Vertex = type %struct.vert_st*
        %struct.graph_st = type { [1 x %Vertex] }
        %struct.hash = type { %HashEntry*, i32 (i32)*, i32 }
        %struct.hash_entry = type { i32, i8*, %HashEntry }
        %struct.vert_st = type { i32, %Vertex, %Hash }
@HashRange = external global i32                ; <i32*> [#uses=0]
@.LC0 = internal global [13 x i8] c"Make phase 2\00"            ; <[13 x i8]*> [#uses=0]
@.LC1 = internal global [13 x i8] c"Make phase 3\00"            ; <[13 x i8]*> [#uses=0]
@.LC2 = internal global [13 x i8] c"Make phase 4\00"            ; <[13 x i8]*> [#uses=0]
@.LC3 = internal global [15 x i8] c"Make returning\00"          ; <[15 x i8]*> [#uses=0]

define %Graph @MakeGraph(i32 %numvert, i32 %numproc) {
bb1:
        %reg111 = add i32 %numproc, -1          ; <i32> [#uses=2]
        %cond275 = icmp slt i32 %reg111, 1              ; <i1> [#uses=1]
        %cond276 = icmp sle i32 %reg111, 0              ; <i1> [#uses=1]
        %cond277 = icmp sge i32 %numvert, 0             ; <i1> [#uses=1]
        %reg162 = add i32 %numvert, 3           ; <i32> [#uses=0]
        br i1 %cond275, label %bb7, label %bb4

bb4:            ; preds = %bb1
        br i1 %cond276, label %bb7, label %bb5

bb5:            ; preds = %bb4
        br i1 %cond277, label %bb7, label %bb6

bb6:            ; preds = %bb5
        ret %Graph null

bb7:            ; preds = %bb5, %bb4, %bb1
        ret %Graph null
}

