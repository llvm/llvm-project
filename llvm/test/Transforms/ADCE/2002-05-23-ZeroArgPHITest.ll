; This testcase contains a entire loop that should be removed.  The only thing
; left is the store instruction in BB0.  The problem this testcase was running
; into was that when the reg109 PHI was getting zero predecessors, it was
; removed even though there were uses still around.  Now the uses are filled
; in with a dummy value before the PHI is deleted.
;
; RUN: opt < %s -S -passes=adce | grep bb1
; RUN: opt < %s -S -passes=adce -adce-remove-loops | FileCheck %s

        %node_t = type { ptr, ptr, ptr, ptr, ptr, i32, i32 }

define void @localize_local(ptr %nodelist) {
bb0:
        %nodelist.upgrd.1 = alloca ptr             ; <ptr> [#uses=2]
        store ptr %nodelist, ptr %nodelist.upgrd.1
        br label %bb1

bb1:            ; preds = %bb0
        %reg107 = load ptr, ptr %nodelist.upgrd.1              ; <ptr> [#uses=2]
        %cond211 = icmp eq ptr %reg107, null               ; <i1> [#uses=1]
; CHECK: br label %bb3
        br i1 %cond211, label %bb3, label %bb2

bb2:            ; preds = %bb2, %bb1
        %reg109 = phi ptr [ %reg110, %bb2 ], [ %reg107, %bb1 ]             ; <ptr> [#uses=1]
        %reg212 = getelementptr %node_t, ptr %reg109, i64 0, i32 1          ; <ptr> [#uses=1]
        %reg110 = load ptr, ptr %reg212                ; <ptr> [#uses=2]
        %cond213 = icmp ne ptr %reg110, null               ; <i1> [#uses=1]
; CHECK: br label %bb3
        br i1 %cond213, label %bb2, label %bb3

bb3:            ; preds = %bb2, %bb1
        ret void
}

