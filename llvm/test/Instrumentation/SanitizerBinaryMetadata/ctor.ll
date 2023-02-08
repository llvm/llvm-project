; RUN: opt < %s -passes='module(sanmd-module)' -sanitizer-metadata-atomics -S | FileCheck %s

; CHECK: $__sanitizer_metadata_atomics.module_ctor = comdat any
; CHECK: $__sanitizer_metadata_atomics.module_dtor = comdat any
; CHECK: $__sanitizer_metadata_covered.module_ctor = comdat any
; CHECK: $__sanitizer_metadata_covered.module_dtor = comdat any

; CHECK: @llvm.used = appending global [4 x ptr] [ptr @__sanitizer_metadata_atomics.module_ctor, ptr @__sanitizer_metadata_atomics.module_dtor, ptr @__sanitizer_metadata_covered.module_ctor, ptr @__sanitizer_metadata_covered.module_dtor], section "llvm.metadata"
; CHECK: @llvm.global_ctors = appending global [2 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 2, ptr @__sanitizer_metadata_atomics.module_ctor, ptr @__sanitizer_metadata_atomics.module_ctor }, { i32, ptr, ptr } { i32 2, ptr @__sanitizer_metadata_covered.module_ctor, ptr @__sanitizer_metadata_covered.module_ctor }]
; CHECK: @llvm.global_dtors = appending global [2 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 2, ptr @__sanitizer_metadata_atomics.module_dtor, ptr @__sanitizer_metadata_atomics.module_dtor }, { i32, ptr, ptr } { i32 2, ptr @__sanitizer_metadata_covered.module_dtor, ptr @__sanitizer_metadata_covered.module_dtor }]

; CHECK: define dso_local void @__sanitizer_metadata_covered.module_ctor() #1 comdat {
; CHECK: define dso_local void @__sanitizer_metadata_covered.module_dtor() #1 comdat {

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i8 @foo(ptr %a) nounwind uwtable {
entry:
  %0 = load atomic i8, ptr %a unordered, align 1
  ret i8 %0
}
