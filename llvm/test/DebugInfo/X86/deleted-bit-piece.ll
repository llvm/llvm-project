; RUN: llc < %s | FileCheck %s
; This is mainly a crasher for the revert in r234717.  A debug info intrinsic
; that gets deleted can't have its bit piece expression verified after it's
; deleted.

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.8.0"

; CHECK: __Z3foov:
; CHECK: retq

define void @_Z3foov() !dbg !12 {
entry:
  br i1 undef, label %exit, label %bb

bb:                                               ; preds = %entry
  call void @llvm.dbg.value(metadata i8* undef, i64 0, metadata !15, metadata !16), !dbg !17
  br label %exit

exit:                                             ; preds = %bb, %entry
  ret void
}

declare void @llvm.dbg.value(metadata, i64, metadata, metadata)

!llvm.module.flags = !{!0, !1}
!llvm.dbg.cu = !{!2}

!0 = !{i32 2, !"Dwarf Version", i32 2}
!1 = !{i32 2, !"Debug Info Version", i32 3}
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, retainedTypes: !5, globals: !4, imports: !4)
!3 = !DIFile(filename: "foo.cpp", directory: "/path/to/dir")
!4 = !{}
!5 = !{!6}
!6 = !DICompositeType(tag: DW_TAG_structure_type, name: "Class", size: 64, align: 64, elements: !7, identifier: "_ZT5Class")
!7 = !{!8, !10}
!8 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !6, baseType: !9, size: 32, align: 32)
!9 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!10 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !6, baseType: !9, size: 32, align: 32)
!12 = distinct !DISubprogram(name: "foo", scope: null, file: !3, type: !13, isLocal: false, isDefinition: true, isOptimized: false, unit: !2)
!13 = !DISubroutineType(types: !14)
!14 = !{null}
!15 = !DILocalVariable(name: "v", scope: !12, type: !6)
!16 = !DIExpression(DW_OP_LLVM_fragment, 32, 32)
!17 = !DILocation(line: 2755, column: 9, scope: !12)
