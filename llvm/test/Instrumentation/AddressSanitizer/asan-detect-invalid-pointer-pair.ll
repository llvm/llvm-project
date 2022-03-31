; RUN: opt < %s -passes='asan-pipeline' -asan-detect-invalid-pointer-cmp -S \
; RUN:     | FileCheck %s --check-prefixes=CMP,NOSUB,ALL
; RUN: opt < %s -passes='asan-pipeline' -asan-detect-invalid-pointer-sub -S \
; RUN:     | FileCheck %s --check-prefixes=SUB,NOCMP,ALL
; RUN: opt < %s -passes='asan-pipeline' -asan-detect-invalid-pointer-pair -S \
; RUN:     | FileCheck %s --check-prefixes=CMP,SUB,ALL
; Support instrumentation of invalid pointer pair detection.

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define i32 @mycmp(i8* %p, i8* %q) sanitize_address {
; ALL-LABEL: @mycmp
; NOCMP-NOT: call void @__sanitizer_ptr_cmp
; CMP: [[P:%[0-9A-Za-z]+]] = ptrtoint i8* %p to i64
; CMP: [[Q:%[0-9A-Za-z]+]] = ptrtoint i8* %q to i64
  %x = icmp ule i8* %p, %q
; CMP: call void @__sanitizer_ptr_cmp(i64 [[P]], i64 [[Q]])
  %y = zext i1 %x to i32
  ret i32 %y
}

define i32 @mysub(i8* %p, i8* %q) sanitize_address {
; ALL-LABEL: @mysub
; NOSUB-NOT: call void @__sanitizer_ptr_sub
; SUB: [[P:%[0-9A-Za-z]+]] = ptrtoint i8* %p to i64
; SUB: [[Q:%[0-9A-Za-z]+]] = ptrtoint i8* %q to i64
  %x = ptrtoint i8* %p to i64
  %y = ptrtoint i8* %q to i64
  %z = sub i64 %x, %y
; SUB: call void @__sanitizer_ptr_sub(i64 [[P]], i64 [[Q]])
  %w = trunc i64 %z to i32
  ret i32 %w
}
