; RUN: llvm-dis < %S/Inputs/coro-retcon-frame-auto-upgrade-test.ll.bc 2>&1 | FileCheck %s 

; ModuleID = '../llvm-project/llvm/test/Transforms/Coroutines/coro-retcon-frame-auto-upgrade.bc'
; CHECK: source_filename = "../llvm-project/llvm/test/Transforms/Coroutines/coro-retcon-frame-auto-upgrade.ll"
; CHECK: target datalayout = "p:64:64:64"

; CHECK: declare void @prototype_f(ptr, i1)

; CHECK: declare noalias ptr @allocate(i32)

; CHECK: declare void @deallocate(ptr)

; CHECK: declare void @init(ptr)

; CHECK: declare void @use(ptr)

; CHECK: declare void @use_addr_val(i64, ptr)

; Function Attrs: presplitcoroutine
; CHECK: define { ptr, ptr } @f(ptr %buffer) #0 {
; CHECK: entry:
; CHECK:   %tmp = alloca { i64, i64 }, align 8
; CHECK:   %proj.1 = getelementptr inbounds { i64, i64 }, ptr %tmp, i64 0, i32 0
; CHECK:   %proj.2 = getelementptr inbounds { i64, i64 }, ptr %tmp, i64 0, i32 1
; CHECK:   store i64 0, ptr %proj.1, align 8
; CHECK:   store i64 0, ptr %proj.2, align 8
; CHECK:   %escape_addr = ptrtoint ptr %tmp to i64
; CHECK:   %id = call token (i32, i32, ptr, ptr, ptr, ptr, ...) @llvm.coro.id.retcon.once(i32 32, i32 8, ptr %buffer, ptr @prototype_f, ptr @allocate, ptr @deallocate)
; CHECK:   %hdl = call ptr @llvm.coro.begin(token %id, ptr null)
; CHECK:   %proj.2.2 = getelementptr inbounds { i64, i64 }, ptr %tmp, i64 0, i32 1
; CHECK:   call void @init(ptr %proj.1)
; CHECK:   call void @init(ptr %proj.2.2)
; CHECK:   call void @use_addr_val(i64 %escape_addr, ptr %tmp)
; CHECK:   %abort = call i1 (...) @llvm.coro.suspend.retcon.i1(ptr %tmp)
; CHECK:   br i1 %abort, label %end, label %resume

; CHECK: resume:                                           ; preds = %entry
; CHECK:   call void @use(ptr %tmp)
; CHECK:   br label %end

; CHECK: end:                                              ; preds = %resume, %entry
; CHECK:   %0 = call i1 @llvm.coro.end(ptr %hdl, i1 false, token none)
; CHECK:   unreachable
; CHECK: }

; Function Attrs: nounwind
; CHECK: declare ptr @llvm.coro.begin(token, ptr writeonly) #1

; Function Attrs: nounwind
; CHECK: declare i1 @llvm.coro.suspend.retcon.i1(...) #1

; Function Attrs: nounwind
; CHECK: declare i1 @llvm.coro.end(ptr, i1, token) #1

; Function Attrs: nounwind
; CHECK: declare token @llvm.coro.id.retcon.once(i32, i32, ptr, ptr, ptr, ptr, ...) #1

; CHECK: attributes #0 = { presplitcoroutine }
; CHECK: attributes #1 = { nounwind }