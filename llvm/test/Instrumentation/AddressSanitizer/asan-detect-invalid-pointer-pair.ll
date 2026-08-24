; RUN: opt < %s -passes=asan -asan-detect-invalid-pointer-cmp -S | FileCheck %s --check-prefixes=CMP,NOSUB,ALL
; RUN: opt < %s -passes=asan -asan-detect-invalid-pointer-sub -S | FileCheck %s --check-prefixes=SUB,NOCMP,ALL
; RUN: opt < %s -passes=asan -asan-detect-invalid-pointer-pair -S | FileCheck %s --check-prefixes=CMP,SUB,ALL
; Support instrumentation of invalid pointer pair detection.

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define i32 @mycmp(ptr %p, ptr %q) sanitize_address {
; ALL-LABEL: @mycmp
; NOCMP-NOT: call void @__sanitizer_ptr_cmp
; CMP: [[P:%[0-9A-Za-z]+]] = ptrtoint ptr %p to i64
; CMP: [[Q:%[0-9A-Za-z]+]] = ptrtoint ptr %q to i64
  %x = icmp ule ptr %p, %q
; CMP: call void @__sanitizer_ptr_cmp(i64 [[P]], i64 [[Q]])
  %y = zext i1 %x to i32
  ret i32 %y
}

define i32 @mysub(ptr %p, ptr %q) sanitize_address {
; ALL-LABEL: @mysub
; NOSUB-NOT: call void @__sanitizer_ptr_sub
; SUB: [[P:%[0-9A-Za-z]+]] = ptrtoint ptr %p to i64
; SUB: [[Q:%[0-9A-Za-z]+]] = ptrtoint ptr %q to i64
  %x = ptrtoint ptr %p to i64
  %y = ptrtoint ptr %q to i64
  %z = sub i64 %x, %y
; SUB: call void @__sanitizer_ptr_sub(i64 [[P]], i64 [[Q]])
  %w = trunc i64 %z to i32
  ret i32 %w
}

define i32 @mysub_ptrtoaddr(ptr %p, ptr %q) sanitize_address {
; ALL-LABEL: @mysub_ptrtoaddr
; NOSUB-NOT: call void @__sanitizer_ptr_sub
; SUB: [[P:%[0-9A-Za-z]+]] = ptrtoaddr ptr %p to i64
; SUB: [[Q:%[0-9A-Za-z]+]] = ptrtoaddr ptr %q to i64
  %x = ptrtoaddr ptr %p to i64
  %y = ptrtoaddr ptr %q to i64
  %z = sub i64 %x, %y
; SUB: call void @__sanitizer_ptr_sub(i64 [[P]], i64 [[Q]])
  %w = trunc i64 %z to i32
  ret i32 %w
}

define <2 x i64> @mysub_vector(<2 x ptr> %p, <2 x ptr> %q) sanitize_address {
; ALL-LABEL: @mysub_vector
; NOSUB-NOT: call void @__sanitizer_ptr_sub
  %x = ptrtoint <2 x ptr> %p to <2 x i64>
  %y = ptrtoint <2 x ptr> %q to <2 x i64>
; SUB: [[P0:%[0-9A-Za-z]+]] = extractelement <2 x i64> %x, i32 0
; SUB: [[Q0:%[0-9A-Za-z]+]] = extractelement <2 x i64> %y, i32 0
; SUB: call void @__sanitizer_ptr_sub(i64 [[P0]], i64 [[Q0]])
; SUB: [[P1:%[0-9A-Za-z]+]] = extractelement <2 x i64> %x, i32 1
; SUB: [[Q1:%[0-9A-Za-z]+]] = extractelement <2 x i64> %y, i32 1
; SUB: call void @__sanitizer_ptr_sub(i64 [[P1]], i64 [[Q1]])
  %z = sub <2 x i64> %x, %y
  ret <2 x i64> %z
}

define <2 x i1> @mycmp_vector(<2 x ptr> %p, <2 x ptr> %q) sanitize_address {
; ALL-LABEL: @mycmp_vector
; NOCMP-NOT: call void @__sanitizer_ptr_cmp
  %x = ptrtoint <2 x ptr> %p to <2 x i64>
  %y = ptrtoint <2 x ptr> %q to <2 x i64>
; CMP: [[P0:%[0-9A-Za-z]+]] = extractelement <2 x i64> %x, i32 0
; CMP: [[Q0:%[0-9A-Za-z]+]] = extractelement <2 x i64> %y, i32 0
; CMP: call void @__sanitizer_ptr_cmp(i64 [[P0]], i64 [[Q0]])
; CMP: [[P1:%[0-9A-Za-z]+]] = extractelement <2 x i64> %x, i32 1
; CMP: [[Q1:%[0-9A-Za-z]+]] = extractelement <2 x i64> %y, i32 1
; CMP: call void @__sanitizer_ptr_cmp(i64 [[P1]], i64 [[Q1]])
  %z = icmp ult <2 x i64> %x, %y
  ret <2 x i1> %z
}

define i32 @mysub_ptrtoint_trunc(ptr %p, ptr %q) sanitize_address {
; ALL-LABEL: @mysub_ptrtoint_trunc
; NOSUB-NOT: call void @__sanitizer_ptr_sub
; SUB: [[P:%[0-9A-Za-z]+]] = ptrtoint ptr %p to i32
; SUB: [[Q:%[0-9A-Za-z]+]] = ptrtoint ptr %q to i32
  %x = ptrtoint ptr %p to i32
  %y = ptrtoint ptr %q to i32
  %z = sub i32 %x, %y
; SUB: [[XP:%[0-9A-Za-z]+]] = zext i32 [[P]] to i64
; SUB: [[XQ:%[0-9A-Za-z]+]] = zext i32 [[Q]] to i64
; SUB: call void @__sanitizer_ptr_sub(i64 [[XP]], i64 [[XQ]])
  ret i32 %z
}

define <2 x i32> @mysub_vector_ptrtoint_trunc(<2 x ptr> %p, <2 x ptr> %q) sanitize_address {
; ALL-LABEL: @mysub_vector_ptrtoint_trunc
; NOSUB-NOT: call void @__sanitizer_ptr_sub
  %x = ptrtoint <2 x ptr> %p to <2 x i32>
  %y = ptrtoint <2 x ptr> %q to <2 x i32>
; SUB: [[X0:%[0-9A-Za-z]+]] = extractelement <2 x i32> %x, i32 0
; SUB: [[ZX0:%[0-9A-Za-z]+]] = zext i32 [[X0]] to i64
; SUB: [[Y0:%[0-9A-Za-z]+]] = extractelement <2 x i32> %y, i32 0
; SUB: [[ZY0:%[0-9A-Za-z]+]] = zext i32 [[Y0]] to i64
; SUB: call void @__sanitizer_ptr_sub(i64 [[ZX0]], i64 [[ZY0]])
; SUB: [[X1:%[0-9A-Za-z]+]] = extractelement <2 x i32> %x, i32 1
; SUB: [[ZX1:%[0-9A-Za-z]+]] = zext i32 [[X1]] to i64
; SUB: [[Y1:%[0-9A-Za-z]+]] = extractelement <2 x i32> %y, i32 1
; SUB: [[ZY1:%[0-9A-Za-z]+]] = zext i32 [[Y1]] to i64
; SUB: call void @__sanitizer_ptr_sub(i64 [[ZX1]], i64 [[ZY1]])
  %z = sub <2 x i32> %x, %y
  ret <2 x i32> %z
}

define i128 @mysub_ptrtoint_widen(ptr %p, ptr %q) sanitize_address {
; ALL-LABEL: @mysub_ptrtoint_widen
; NOSUB-NOT: call void @__sanitizer_ptr_sub
; SUB: [[P:%[0-9A-Za-z]+]] = ptrtoint ptr %p to i128
; SUB: [[Q:%[0-9A-Za-z]+]] = ptrtoint ptr %q to i128
  %x = ptrtoint ptr %p to i128
  %y = ptrtoint ptr %q to i128
; SUB-NOT: zext i128
; SUB: [[XP:%[0-9A-Za-z]+]] = trunc i128 [[P]] to i64
; SUB: [[XQ:%[0-9A-Za-z]+]] = trunc i128 [[Q]] to i64
; SUB: call void @__sanitizer_ptr_sub(i64 [[XP]], i64 [[XQ]])
  %z = sub i128 %x, %y
  ret i128 %z
}
