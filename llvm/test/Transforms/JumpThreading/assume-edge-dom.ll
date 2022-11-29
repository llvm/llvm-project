; RUN: opt -S -passes=jump-threading < %s | FileCheck %s

declare ptr @escape()
declare void @llvm.assume(i1)

define i1 @test1(i1 %cond) {
entry:
    br i1 %cond, label %taken, label %not_taken

; CHECK-LABEL: @test1
; CHECK: br i1 %cond, label %no, label %yes
; CHECK: ret i1 true

taken:
    %res1 = call ptr @escape()
    %a = icmp eq ptr %res1, null
    tail call void @llvm.assume(i1 %a)
    br label %done
not_taken:
    %res2 = call ptr @escape()
    %b = icmp ne ptr %res2, null
    tail call void @llvm.assume(i1 %b)
    br label %done

; An assume that can be used to simplify this comparison dominates each
; predecessor branch (although no assume dominates the cmp itself). Make sure
; this still can be simplified.

done:
    %res = phi ptr [ %res1, %taken ], [ %res2, %not_taken ]
    %cnd = icmp ne ptr %res, null
    br i1 %cnd, label %yes, label %no

yes:
    ret i1 true
no:
    ret i1 false
}

