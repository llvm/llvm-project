; RUN: llc -global-isel -mtriple=aarch64 %s -stop-after=irtranslator -o - | FileCheck %s


; CHECK: ![[var_dbg_declare:[0-9]+]] = !DILocalVariable(name: "var1"
; CHECK: ![[var_dbg_value:[0-9]+]] = !DILocalVariable(name: "var2"


define void @debug_declare(i8* swiftasync %async_state) !dbg !7 {
  call void @llvm.dbg.declare(metadata i8* %async_state, metadata !11, metadata !DIExpression(DW_OP_plus_uconst, 16, DW_OP_plus_uconst, 16)), !dbg !12
  ; CHECK: DBG_VALUE $x22, 0, ![[var_dbg_declare]], !DIExpression(DW_OP_plus_uconst, 16, DW_OP_plus_uconst, 16)
  ret void, !dbg !12
}

define void @debug_value(i8* swiftasync %async_state) !dbg !13 {
  call void @llvm.dbg.value(metadata i8* %async_state, metadata !14, metadata !DIExpression(DW_OP_plus_uconst, 16, DW_OP_plus_uconst, 16, DW_OP_deref)), !dbg !15
  ; CHECK: DBG_VALUE $x22, $noreg, ![[var_dbg_value]], !DIExpression(DW_OP_plus_uconst, 16, DW_OP_plus_uconst, 16, DW_OP_deref)
  ret void, !dbg !15
}

declare void @llvm.dbg.declare(metadata, metadata, metadata)
declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "myclang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "tmp.c", directory: "blah")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}

!8 = !DISubroutineType(types: !9)
!9 = !{null, !10}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)

!7 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 1, type: !8, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!13 = distinct !DISubprogram(name: "foo2", scope: !1, file: !1, line: 1, type: !8, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)

!11 = !DILocalVariable(name: "var1", arg: 1, scope: !7, file: !1, line: 1, type: !10)
!14 = !DILocalVariable(name: "var2", arg: 1, scope: !13, file: !1, line: 1, type: !10)

!12 = !DILocation(line: 1, column: 14, scope: !7)
!15 = !DILocation(line: 1, column: 14, scope: !13)
