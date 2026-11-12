; RUN: llc  -verify-machineinstrs < %s | FileCheck %s

; Used to crash with assertions when emitting object files.
; RUN: llc -filetype=obj %s -o /dev/null

; Check stack map section is emitted before debug info.
; CHECK: .section __LLVM_STACKMAPS
; CHECK: .section __DWARF

target triple = "x86_64-apple-macosx12.0"

declare void @func()

define ptr addrspace(1) @test1(ptr addrspace(1) %arg) gc "statepoint-example" {
entry:
  %safepoint_token = call token (i64, i32, ptr, i32, i32, ...) @llvm.experimental.gc.statepoint.p0(i64 0, i32 0, ptr elementtype(void ()) @func, i32 0, i32 0, i32 0, i32 0) ["gc-live"(ptr addrspace(1) %arg)]
  %reloc1 = call ptr addrspace(1) @llvm.experimental.gc.relocate.p1(token %safepoint_token,  i32 0, i32 0)
  ret ptr addrspace(1) %reloc1
}

declare token @llvm.experimental.gc.statepoint.p0(i64, i32, ptr, i32, i32, ...)
declare ptr addrspace(1) @llvm.experimental.gc.relocate.p1(token, i32, i32)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}
!0 = distinct !DICompileUnit(language: DW_LANG_C89, producer: "clang", isOptimized: true, emissionKind: FullDebug, file: !1, enums: !{}, retainedTypes: !{})
!1 = !DIFile(filename: "t.c", directory: "")
!2 = !{i32 2, !"Debug Info Version", i32 3}
