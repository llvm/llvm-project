; RUN: llc -mtriple=pisa -filetype=asm %s -o - | FileCheck %s

target triple = "pisa"

@metadata_string = private addrspace(2) constant [5 x i8] c"meta\00", section "llvm.metadata"
@ordinary_string = private addrspace(2) constant [6 x i8] c"plain\00"
@data = addrspace(1) global i32 42

define ptr @annotate(ptr %p) {
  %annotated = call ptr @llvm.ptr.annotation.p0.p2(ptr %p, ptr addrspace(2) @metadata_string, ptr addrspace(2) @ordinary_string, i32 1, ptr addrspace(2) null)
  ret ptr %annotated
}

declare ptr @llvm.ptr.annotation.p0.p2(ptr, ptr addrspace(2), ptr addrspace(2), i32, ptr addrspace(2))

; CHECK-NOT: @metadata_string
; CHECK: @ordinary_string =
; CHECK: @data =
; CHECK-NOT: @metadata_string
