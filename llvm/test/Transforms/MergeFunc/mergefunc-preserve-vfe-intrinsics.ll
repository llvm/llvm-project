; This test contains three identical functions, aside from the metadata
; they pass to a function call. This test verifies that the function merger
; pass is able to merge the two functions that are truly identifical,
; but the third that passes different metadata is preserved

; RUN: opt -passes=mergefunc -S %s | FileCheck %s

declare { ptr, i1 } @llvm.type.checked.load(ptr, i32, metadata)

define i1 @merge_candidate_a(ptr %ptr, i32 %offset) {
    %1 = call { ptr, i1 } @llvm.type.checked.load(ptr %ptr, i32 %offset, metadata !"common_metadata")
    %2 = extractvalue { ptr, i1 } %1, 1
    ret i1 %2
}

define i1 @merge_candidate_c(ptr %ptr, i32 %offset) {
    %1 = call { ptr, i1 } @llvm.type.checked.load(ptr %ptr, i32 %offset, metadata !"different_metadata")
    %2 = extractvalue { ptr, i1 } %1, 1
    ret i1 %2
}
; CHECK-LABEL: @merge_candidate_c
; CHECK-NOT: call i1 merge_candidate_a
; CHECK: call { ptr, i1 } @llvm.type.checked.load
; CHECK-NOT: call i1 merge_candidate_a
; CHECK: ret

define i1 @merge_candidate_b(ptr %ptr, i32 %offset) {
    %1 = call { ptr, i1 } @llvm.type.checked.load(ptr %ptr, i32 %offset, metadata !"common_metadata")
    %2 = extractvalue { ptr, i1 } %1, 1
    ret i1 %2
}
; CHECK-LABEL: @merge_candidate_b
; CHECK: call i1 @merge_candidate_a
; CHECK: ret
