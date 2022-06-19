; This testcase contains a entire loop that should be removed.  The only thing
; left is the store instruction in BB0.  The problem this testcase was running
; into was that when the reg109 PHI was getting zero predecessors, it was
; removed even though there were uses still around.  Now the uses are filled
; in with a dummy value before the PHI is deleted.
;
; RUN: opt < %s -S -passes=adce | grep bb1
; RUN: opt < %s -S -passes=adce -adce-remove-loops | FileCheck %s

        %node_t = type { double*, %node_t*, %node_t**, double**, double*, i32, i32 }

define void @localize_local(%node_t* %nodelist) {
bb0:
        %nodelist.upgrd.1 = alloca %node_t*             ; <%node_t**> [#uses=2]
        store %node_t* %nodelist, %node_t** %nodelist.upgrd.1
        br label %bb1

bb1:            ; preds = %bb0
        %reg107 = load %node_t*, %node_t** %nodelist.upgrd.1              ; <%node_t*> [#uses=2]
        %cond211 = icmp eq %node_t* %reg107, null               ; <i1> [#uses=1]
; CHECK: br label %bb3
        br i1 %cond211, label %bb3, label %bb2

bb2:            ; preds = %bb2, %bb1
        %reg109 = phi %node_t* [ %reg110, %bb2 ], [ %reg107, %bb1 ]             ; <%node_t*> [#uses=1]
        %reg212 = getelementptr %node_t, %node_t* %reg109, i64 0, i32 1          ; <%node_t**> [#uses=1]
        %reg110 = load %node_t*, %node_t** %reg212                ; <%node_t*> [#uses=2]
        %cond213 = icmp ne %node_t* %reg110, null               ; <i1> [#uses=1]
; CHECK: br label %bb3
        br i1 %cond213, label %bb2, label %bb3

bb3:            ; preds = %bb2, %bb1
        ret void
}

