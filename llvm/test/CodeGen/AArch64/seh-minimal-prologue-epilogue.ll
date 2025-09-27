; RUN: llc -mtriple=aarch64-windows %s -o - | FileCheck %s

; This test verifies that functions requiring Windows CFI that have minimal
; or no prologue instructions still emit proper SEH directives, specifically
; ensuring .seh_endprologue is emitted before .seh_startepilogue.
;
; This reproduces the issue where Swift async functions with swifttailcc
; calling convention would fail with:
; "error: starting epilogue (.seh_startepilogue) before prologue has ended (.seh_endprologue)"

; Test 1: Swift-style tail call function with minimal prologue
define swifttailcc void @test_swifttailcc_minimal(ptr %async_ctx, ptr %arg1, ptr %arg2) {
; CHECK-LABEL: test_swifttailcc_minimal:
; CHECK-NOT:   .seh_proc test_swifttailcc_minimal
; CHECK-NOT:   .seh_endprologue
; CHECK-NOT:   .seh_startepilogue
; CHECK-NOT:   .seh_endepilogue
; CHECK-NOT:   .seh_endproc
entry:
  %ptr1 = getelementptr inbounds i8, ptr %async_ctx, i64 16
  %ptr2 = getelementptr inbounds i8, ptr %async_ctx, i64 24
  store ptr %arg1, ptr %ptr1, align 8
  store ptr %arg2, ptr %ptr2, align 8
  musttail call swifttailcc void @external_swift_function(ptr %async_ctx, ptr %arg1)
  ret void
}

; Test 2: Regular function with no stack frame but needs epilogue
define void @test_no_stack_frame() {
; CHECK-LABEL: test_no_stack_frame:
; CHECK-NEXT:  .seh_proc test_no_stack_frame
; CHECK:       .seh_endprologue
; CHECK:       .seh_startepilogue
; CHECK:       .seh_endepilogue
; CHECK:       .seh_endproc
entry:
  call void @external_function()
  ret void
}

; Test 3: Function with minimal stack adjustment only in epilogue
define void @test_minimal_stack_adjust(ptr %ptr) {
; CHECK-LABEL: test_minimal_stack_adjust:
; CHECK-NEXT:  .seh_proc test_minimal_stack_adjust
; CHECK:       .seh_endprologue
; CHECK:       .seh_startepilogue
; CHECK:       add sp, sp, #16
; CHECK:       .seh_stackalloc 16
; CHECK:       .seh_endepilogue
; CHECK:       .seh_endproc
entry:
  %local = alloca i64, align 8
  store i64 42, ptr %local, align 8
  %value = load i64, ptr %local, align 8
  store i64 %value, ptr %ptr, align 8
  ret void
}

; Test 4: Function similar to the original failing case
define linkonce_odr hidden swifttailcc void @test_linkonce_swifttailcc(ptr swiftasync %async_ctx, ptr %arg1, ptr noalias dereferenceable(40) %arg2, ptr %arg3, i64 %value, ptr %arg4, ptr %arg5, ptr %arg6, i1 %flag, ptr %arg7, ptr noalias dereferenceable(40) %arg8) {
; CHECK-LABEL: test_linkonce_swifttailcc:
; CHECK-NEXT:  .seh_proc
; CHECK:       .seh_endprologue
; CHECK:       .seh_startepilogue
; CHECK:       .seh_endepilogue
; CHECK:       .seh_endproc
entry:
  %frame_ptr = getelementptr inbounds nuw i8, ptr %async_ctx, i64 16
  %ctx1 = getelementptr inbounds nuw i8, ptr %async_ctx, i64 400
  %ctx2 = getelementptr inbounds nuw i8, ptr %async_ctx, i64 1168
  %spill1 = getelementptr inbounds nuw i8, ptr %async_ctx, i64 2392
  store ptr %arg8, ptr %spill1, align 8
  %spill2 = getelementptr inbounds nuw i8, ptr %async_ctx, i64 2384
  store ptr %arg7, ptr %spill2, align 8
  %spill3 = getelementptr inbounds nuw i8, ptr %async_ctx, i64 2225
  store i1 %flag, ptr %spill3, align 1
  %spill4 = getelementptr inbounds nuw i8, ptr %async_ctx, i64 2376
  store ptr %arg6, ptr %spill4, align 8
  musttail call swifttailcc void @external_swift_continuation(ptr swiftasync %async_ctx, i64 0, i64 0)
  ret void
}

declare swifttailcc void @external_swift_function(ptr, ptr)
declare swifttailcc void @external_swift_continuation(ptr swiftasync, i64, i64)
declare void @external_function()
