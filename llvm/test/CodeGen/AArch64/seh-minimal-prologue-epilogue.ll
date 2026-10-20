; RUN: llc -mtriple=aarch64-windows %s -o - | FileCheck %s --check-prefixes=CHECK,WIN
; RUN: llc -mtriple=arm64-apple-ios %s -o - | FileCheck %s --check-prefixes=CHECK,IOS --implicit-check-not=.seh

; This test verifies that functions requiring Windows CFI that have minimal
; or no prologue instructions still emit proper SEH directives, specifically
; ensuring .seh_endprologue is emitted before .seh_startepilogue.
;
; This reproduces the issue where Swift async functions with swifttailcc
; calling convention would fail with:
; "error: starting epilogue (.seh_startepilogue) before prologue has ended (.seh_endprologue)"

; Swift-style tail call function with minimal prologue
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

; Function similar to the original failing case
define linkonce_odr hidden swifttailcc void @test_linkonce_swifttailcc(ptr swiftasync %async_ctx, ptr %arg1, ptr noalias dereferenceable(40) %arg2, ptr %arg3, i64 %value, ptr %arg4, ptr %arg5, ptr %arg6, i1 %flag, ptr %arg7, ptr noalias dereferenceable(40) %arg8) {
; CHECK-LABEL: test_linkonce_swifttailcc:
; IOS-NEXT:  .cfi_startproc
; WIN-NEXT:  .seh_proc
; WIN:       .seh_endprologue
; WIN:       .seh_startepilogue
; WIN:       .seh_endepilogue
; WIN:       .seh_endproc
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
