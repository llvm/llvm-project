; RUN: opt -O0 -S -mtriple=amdgcn-unknown-amdhsa -amdgpu-internalize-symbols < %s | FileCheck -check-prefix=OPTNONE %s
; RUN: opt -passes='default<O0>' -S -mtriple=amdgcn-unknown-amdhsa -amdgpu-internalize-symbols < %s | FileCheck -check-prefix=OPTNONE %s
; RUN: opt -O1 -S -mtriple=amdgcn-unknown-amdhsa -amdgpu-internalize-symbols < %s | FileCheck -check-prefix=ASAN_NO_INTERNALIZE %s
; RUN: opt -passes='default<O1>' -S -mtriple=amdgcn-unknown-amdhsa -amdgpu-internalize-symbols < %s | FileCheck -check-prefix=ASAN_NO_INTERNALIZE %s

; OPTNONE: define void @__asan_no_explicit_linkage(
; ASAN_NO_INTERNALIZE: define void @__asan_no_explicit_linkage(
define void @__asan_no_explicit_linkage() {
entry:
  ret void
}

; OPTNONE: define weak void @__asan_weak_linkage(
; ASAN_NO_INTERNALIZE: define weak void @__asan_weak_linkage(
define weak void @__asan_weak_linkage() {
entry:
  ret void
}

; OPTNONE: define void @__sanitizer_no_explicit_linkage(
; ASAN_NO_INTERNALIZE: define void @__sanitizer_no_explicit_linkage(
define void @__sanitizer_no_explicit_linkage() {
entry:
  ret void
}

; OPTNONE: define weak void @__sanitizer_weak_linkage(
; ASAN_NO_INTERNALIZE: define weak void @__sanitizer_weak_linkage(
define weak void @__sanitizer_weak_linkage() {
entry:
  ret void
}
