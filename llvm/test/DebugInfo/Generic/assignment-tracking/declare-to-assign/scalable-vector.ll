; RUN: opt -passes=declare-to-assign %s -S | FileCheck %s
; RUN: opt --try-experimental-debuginfo-iterators -passes=declare-to-assign %s -S | FileCheck %s

;; Check declare-to-assign skips scalable vectors for now. i.e. do not replace
;; the dbg.declare with a dbg.assign intrinsic.

; CHECK: #dbg_declare(ptr %c

define dso_local void @b() !dbg !9 {
entry:
  %c = alloca <vscale x 8 x i32>, align 4
  call void @llvm.dbg.declare(metadata ptr %c, metadata !13, metadata !DIExpression()), !dbg !21
  ret void
}

declare void @llvm.dbg.declare(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !5, !7}
!llvm.ident = !{!8}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang version 17.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.c", directory: "/")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"target-abi", !"lp64"}
!7 = !{i32 8, !"SmallDataLimit", i32 8}
!8 = !{!"clang version 17.0.0"}
!9 = distinct !DISubprogram(name: "b", scope: !1, file: !1, line: 2, type: !10, scopeLine: 2, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !12)
!10 = !DISubroutineType(types: !11)
!11 = !{null}
!12 = !{}
!13 = !DILocalVariable(name: "c", scope: !9, file: !1, line: 2, type: !14)
!14 = !DIDerivedType(tag: DW_TAG_typedef, name: "a", file: !1, line: 1, baseType: !15)
!15 = !DIDerivedType(tag: DW_TAG_typedef, name: "__rvv_uint32m4_t", file: !16, baseType: !17)
!16 = !DIFile(filename: "test.c", directory: "/")
!17 = !DICompositeType(tag: DW_TAG_array_type, baseType: !18, flags: DIFlagVector, elements: !19)
!18 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!19 = !{!20}
!20 = !DISubrange(lowerBound: 0, upperBound: !DIExpression(DW_OP_bregx, 7202, 0, DW_OP_constu, 4, DW_OP_div, DW_OP_constu, 4, DW_OP_mul, DW_OP_constu, 1, DW_OP_minus))
!21 = !DILocation(line: 2, column: 14, scope: !9)

