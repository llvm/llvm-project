; RUN: opt -S -debugify -codegenprepare %s -o - | FileCheck %s --check-prefix=DEBUGIFY
; RUN: opt -S -debugify -codegenprepare %s -o - --try-experimental-debuginfo-iterators | FileCheck %s --check-prefix=DEBUGIFY
;
; Copied from codegen-prepare-addrmode-sext.ll -- for the twoArgsNoPromotion
; function, CGP attempts a type promotion transaction on the sext to replace
; it with %add, but then rolls it back. This involves re-inserting the sext
; instruction between two dbg.value intrinsics, and un-RAUWing the users of
; the sext.
; This test checks that this works correctly in both dbg.value mode, but also
; RemoveDIs non-intrinsic debug-info mode.

target datalayout = "e-i64:64-f80:128-s:64-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx"

; DEBUGIFY-LABEL: @twoArgsNoPromotion
; DEBUGIFY-NEXT: %add = add
; DEBUGIFY-NEXT: #dbg_value(i32 %add,
; DEBUGIFY-NEXT: %sextadd = sext
; DEBUGIFY-NEXT: #dbg_value(i64 %sextadd,
; DEBUGIFY-NEXT: %arrayidx = getelementptr
; DEBUGIFY-NEXT: #dbg_value(ptr %arrayidx,
; DEBUGIFY-NEXT: %res = load i8,
; DEBUGIFY-NEXT: #dbg_value(i8 %res,
; DEBUGIFY-NEXT: ret i8 %res,
define i8 @twoArgsNoPromotion(i32 %arg1, i32 %arg2, ptr %base) {
  %add = add nsw i32 %arg1, %arg2
  %sextadd = sext i32 %add to i64
  %arrayidx = getelementptr inbounds i8, ptr %base, i64 %sextadd
  %res = load i8, ptr %arrayidx
  ret i8 %res
}

