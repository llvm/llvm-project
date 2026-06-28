; Test sanitizer coverage trace-ret instrumentation.
; Verifies that __sanitizer_cov_trace_ret is called for struct pointer and scalar returns.

; RUN: opt < %s -passes='module(sancov-module)' -sanitizer-coverage-level=3 -sanitizer-coverage-trace-ret -S | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.MyStruct = type { i32, i64 }

define ptr @func_ret_struct_ptr(ptr %s) #0 !dbg !8 {
entry:
  ret ptr %s
}

; CHECK: define ptr @func_ret_struct_ptr(ptr %s)
; CHECK: call void @__sanitizer_cov_trace_ret(i64 ptrtoint (ptr @func_ret_struct_ptr to i64), i32 8, ptr %s, ptr getelementptr inbounds ([5 x i64], ptr @__sancov_offsets_{{.*}}, i64 0, i64 1), i32 2)
; CHECK: ret ptr %s

define i32 @func_ret_scalar(i32 %x) #0 !dbg !15 {
entry:
  ret i32 %x
}

; CHECK: define i32 @func_ret_scalar(i32 %x)
; CHECK: call void @__sanitizer_cov_trace_ret(i64 ptrtoint (ptr @func_ret_scalar to i64), i32 4, ptr %{{.*}}, ptr null, i32 0)
; CHECK: ret i32 %x

attributes #0 = { nounwind sanitize_address }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, isOptimized: false, emissionKind: FullDebug)
!1 = !DIFile(filename: "test.c", directory: "/tmp")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}

; struct MyStruct { int a; long b; }
!5 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!6 = !DIBasicType(name: "long", size: 64, encoding: DW_ATE_signed)
!7 = !DICompositeType(tag: DW_TAG_structure_type, name: "MyStruct", size: 128, elements: !14)

; func_ret_struct_ptr returns ptr to MyStruct
!8 = distinct !DISubprogram(name: "func_ret_struct_ptr", scope: !1, file: !1, line: 5, type: !9, unit: !0, retainedNodes: !2)
!9 = !DISubroutineType(types: !10)
; types: [ret=ptr to MyStruct, arg0=ptr to MyStruct]
!10 = !{!11, !11}
!11 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !7, size: 64)
!12 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !7, file: !1, baseType: !5, size: 32, offset: 0)
!13 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !7, file: !1, baseType: !6, size: 64, offset: 64)
!14 = !{!12, !13}

; func_ret_scalar returns i32
!15 = distinct !DISubprogram(name: "func_ret_scalar", scope: !1, file: !1, line: 10, type: !16, unit: !0, retainedNodes: !2)
!16 = !DISubroutineType(types: !17)
; types: [ret=int, arg0=int]
!17 = !{!5, !5}
