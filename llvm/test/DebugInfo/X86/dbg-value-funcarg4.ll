; RUN: llc -mtriple=x86_64-unknown-linux-gnu -start-after=codegenprepare -stop-before=finalize-isel -o - %s -experimental-debug-variable-locations=true | FileCheck %s --implicit-check-not=DBG_PHI

; Test that for multiple dbg.values referring to the same argument, we emit a
; single DBG_PHI and refer to it twice. (Using more than one DBG_PHI is fine,
; but inefficient).

; CHECK-DAG: ![[LOCAL:.*]] = !DILocalVariable(name: "local"
; CHECK-DAG: ![[LOCAL2:.*]] = !DILocalVariable(name: "local2"

; CHECK: DBG_PHI $edi, 1

; CHECK: DBG_INSTR_REF ![[LOCAL]], !DIExpression(DW_OP_LLVM_arg, 0), dbg-instr-ref(1, 0)
; CHECK: DBG_INSTR_REF ![[LOCAL2]], !DIExpression(DW_OP_LLVM_arg, 0), dbg-instr-ref(1, 0)

declare void @bar(i32)
declare void @llvm.dbg.value(metadata, metadata, metadata)

define dso_local void @foo_local(i32 %t1a) local_unnamed_addr !dbg !7 {
entry:
  tail call void @bar(i32 %t1a) #3, !dbg !17
  %bees = add i32 %t1a, 3
  call void @llvm.dbg.value(metadata i32 %t1a, metadata !13, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.value(metadata i32 %t1a, metadata !19, metadata !DIExpression()), !dbg !14
  tail call void @bar(i32 %bees) #3, !dbg !17
  ret void, !dbg !18
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "foo.c", directory: "")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang"}
!7 = distinct !DISubprogram(name: "foo_local", scope: !1, file: !1, line: 3, type: !8, scopeLine: 3, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !11)
!8 = !DISubroutineType(types: !9)
!9 = !{null, !10}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !{!12, !13}
!12 = !DILocalVariable(name: "t1a", arg: 1, scope: !7, file: !1, line: 3, type: !10)
!13 = !DILocalVariable(name: "local", scope: !7, file: !1, line: 4, type: !10)
!14 = !DILocation(line: 3, column: 20, scope: !7)
!15 = !DILocation(line: 4, column: 7, scope: !7)
!16 = !DILocation(line: 5, column: 3, scope: !7)
!17 = !DILocation(line: 7, column: 3, scope: !7)
!18 = !DILocation(line: 8, column: 1, scope: !7)
!19 = !DILocalVariable(name: "local2", scope: !7, file: !1, line: 4, type: !10)
