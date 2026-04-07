; RUN: mlir-translate -import-llvm -mlir-print-debuginfo %s | FileCheck %s

; CHECK-DAG: #[[CU_SELF:.+]] = #llvm.di_compile_unit<{{.*}}recId = distinct[{{[0-9]+}}]<>{{.*}}isRecSelf = true{{.*}}>
; CHECK-DAG: #llvm.di_imported_entity<{{.*}}tag = DW_TAG_imported_declaration{{.*}}scope = #[[CU_SELF]]{{.*}}>
; CHECK-DAG: #llvm.di_compile_unit<{{.*}}recId = distinct[{{[0-9]+}}]<>{{.*}}id = distinct[{{[0-9]+}}]<>{{.*}}importedEntities{{.*}}>

source_filename = "compile-unit-imported-entity-cycle.ll"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f64:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@g = external global i32, !dbg !0

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!17}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "g", scope: !2, file: !12, line: 7, type: !13, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !3, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, retainedTypes: !4, globals: !4, imports: !5, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "cu.cpp", directory: "/build")
!4 = !{}
!5 = !{!6}
!6 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !2, entity: !7, file: !11, line: 10)
!7 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "S", scope: !9, file: !8, line: 5, size: 32, flags: DIFlagTypePassByValue, elements: !4)
!8 = !DIFile(filename: "hdr.hpp", directory: "/build")
!9 = !DINamespace(name: "ns", scope: !10)
!10 = !DINamespace(name: "outer", scope: null)
!11 = !DIFile(filename: "import.hpp", directory: "/build")
!12 = !DIFile(filename: "cu.hpp", directory: "/build")
!13 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!17 = !{i32 2, !"Debug Info Version", i32 3}
