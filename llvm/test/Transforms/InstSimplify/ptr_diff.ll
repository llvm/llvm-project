; RUN: opt < %s -instsimplify -S | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i64 @ptrdiff1(i8* %ptr) {
; CHECK: @ptrdiff1
; CHECK-NEXT: ret i64 42

  %first = getelementptr inbounds i8* %ptr, i32 0
  %last = getelementptr inbounds i8* %ptr, i32 42
  %first.int = ptrtoint i8* %first to i64
  %last.int = ptrtoint i8* %last to i64
  %diff = sub i64 %last.int, %first.int
  ret i64 %diff
}

define i64 @ptrdiff2(i8* %ptr) {
; CHECK: @ptrdiff2
; CHECK-NEXT: ret i64 42

  %first1 = getelementptr inbounds i8* %ptr, i32 0
  %first2 = getelementptr inbounds i8* %first1, i32 1
  %first3 = getelementptr inbounds i8* %first2, i32 2
  %first4 = getelementptr inbounds i8* %first3, i32 4
  %last1 = getelementptr inbounds i8* %first2, i32 48
  %last2 = getelementptr inbounds i8* %last1, i32 8
  %last3 = getelementptr inbounds i8* %last2, i32 -4
  %last4 = getelementptr inbounds i8* %last3, i32 -4
  %first.int = ptrtoint i8* %first4 to i64
  %last.int = ptrtoint i8* %last4 to i64
  %diff = sub i64 %last.int, %first.int
  ret i64 %diff
}

define i64 @ptrdiff3(i8* %ptr) {
; Don't bother with non-inbounds GEPs.
; CHECK: @ptrdiff3
; CHECK: getelementptr
; CHECK: sub
; CHECK: ret

  %first = getelementptr i8* %ptr, i32 0
  %last = getelementptr i8* %ptr, i32 42
  %first.int = ptrtoint i8* %first to i64
  %last.int = ptrtoint i8* %last to i64
  %diff = sub i64 %last.int, %first.int
  ret i64 %diff
}
