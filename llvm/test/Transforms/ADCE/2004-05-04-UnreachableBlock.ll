; RUN: opt < %s -adce -disable-output

define void @test() {
entry:
        br label %UnifiedReturnBlock

UnifiedReturnBlock:             ; preds = %invoke_catch.0, %entry
        ret void

invoke_catch.0:         ; No predecessors!
        br i1 false, label %UnifiedUnwindBlock, label %UnifiedReturnBlock

UnifiedUnwindBlock:             ; preds = %invoke_catch.0
        unreachable
}

