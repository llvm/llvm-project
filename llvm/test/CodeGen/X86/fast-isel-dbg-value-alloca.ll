; RUN: llc -fast-isel -fast-isel-abort=1 -mtriple=x86_64-unknown-unknown -stop-after=finalize-isel %s -o - | \
; RUN:    FileCheck %s

; RUN: llc --try-experimental-debuginfo-iterators -fast-isel -fast-isel-abort=1 -mtriple=x86_64-unknown-unknown -stop-after=finalize-isel %s -o - |  FileCheck %s

define void @foo(ptr noalias nocapture %arg) !dbg !38 {
  %k.debug = alloca ptr, align 8
  store ptr %arg, ptr %k.debug, align 8, !dbg !70
  call void @llvm.dbg.value(metadata ptr %k.debug, metadata !55, metadata !DIExpression(DW_OP_deref)), !dbg !70
; CHECK: #dbg_value(ptr %{{.*}}, ![[VAR:.*]], ![[EXPR:.*]], 
; CHECK: DBG_VALUE %stack.0{{.*}}, $noreg, ![[VAR]], ![[EXPR]]
  ret void, !dbg !70
}

declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.module.flags = !{!6, !7, !8, !9}
!llvm.dbg.cu = !{!16}

!6 = !{i32 7, !"Dwarf Version", i32 4}
!7 = !{i32 2, !"Debug Info Version", i32 3}
!8 = !{i32 1, !"wchar_size", i32 4}
!9 = !{i32 8, !"PIC Level", i32 2}
!16 = distinct !DICompileUnit(language: DW_LANG_Swift, file: !17, producer: "blah", isOptimized: false, runtimeVersion: 5, emissionKind: FullDebug, sysroot: "blah", sdk: "blah")
!17 = !DIFile(filename: "blah", directory: "blah")
!38 = distinct !DISubprogram(name: "blah", linkageName: "$blah", scope: !17, file: !17, line: 34, type: !39, scopeLine: 34, spFlags: DISPFlagDefinition, unit: !16, retainedNodes: !43)
!39 = !DISubroutineType(types: !40)
!40 = !{!41, !41}
!41 = !DICompositeType(tag: DW_TAG_structure_type, name: "blah")
!43 = !{!49, !55}
!49 = !DILocalVariable(name: "x", arg: 1, scope: !38, file: !17, line: 34, type: !41)
!55 = !DILocalVariable(name: "k", scope: !56, file: !17, line: 36, type: !41)
!56 = distinct !DILexicalBlock(scope: !38, file: !17, line: 36, column: 9)
!70 = !DILocation(line: 36, column: 9, scope: !56)
