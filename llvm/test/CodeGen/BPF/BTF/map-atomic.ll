; RUN: llc -mtriple=bpfel %s -o - | FileCheck %s
; RUN: llc -mtriple=bpfeb %s -o - | FileCheck %s
; CHECK: BTF_KIND_FWD
; CHECK-NOT: BTF_KIND_STRUCT
; CHECK: .ascii "Leaf"
; ModuleID = 'map-atomic.c'
source_filename = "map-atomic.c"
target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128"
target triple = "bpfel"

%struct.Wrapper = type { %struct.Inner }
%struct.Inner = type { ptr }

@m = dso_local global %struct.Wrapper zeroinitializer, section ".maps", align 8, !dbg !0

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!17, !18, !19, !20}
!llvm.ident = !{!21}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "m", scope: !2, file: !3, line: 4, type: !5, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C11, file: !3, producer: "clang version 24.0.0git (git@github.com:caniko/llvm-project.git cd014203541ab1ec3cd7caed62e64ba403773d00)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "map-atomic.c", directory: "/")
!4 = !{!0}
!5 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Wrapper", file: !3, line: 3, size: 64, elements: !6)
!6 = !{!7}
!7 = !DIDerivedType(tag: DW_TAG_member, name: "member", scope: !5, file: !3, line: 3, baseType: !8, size: 64)
!8 = !DIDerivedType(tag: DW_TAG_atomic_type, baseType: !9)
!9 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Inner", file: !3, line: 2, size: 64, elements: !10)
!10 = !{!11}
!11 = !DIDerivedType(tag: DW_TAG_member, name: "p", scope: !9, file: !3, line: 2, baseType: !12, size: 64)
!12 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !13, size: 64)
!13 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Leaf", file: !3, line: 1, size: 32, elements: !14)
!14 = !{!15}
!15 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !13, file: !3, line: 1, baseType: !16, size: 32)
!16 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!17 = !{i32 7, !"Dwarf Version", i32 5}
!18 = !{i32 2, !"Debug Info Version", i32 3}
!19 = !{i32 2, !"debugger-tuning", !"gdb"}
!20 = !{i32 7, !"frame-pointer", i32 2}
!21 = !{!"clang version 24.0.0git (git@github.com:caniko/llvm-project.git cd014203541ab1ec3cd7caed62e64ba403773d00)"}
