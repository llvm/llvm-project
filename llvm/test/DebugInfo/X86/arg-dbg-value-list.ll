;; FIXME: We currently don't do any special bookkeeping for unused args used by
;; variadic dbg_values. When/if we support them, the DBG_VALUE_LIST should be
;; updated accordingly.

; RUN: llc %s -start-after=codegenprepare --experimental-debug-variable-locations=false -stop-before=finalize-isel -o - | FileCheck %s
; RUN: llc %s -start-after=codegenprepare --experimental-debug-variable-locations -stop-before=finalize-isel -o - | FileCheck %s

; RUN: llc %s -start-after=codegenprepare --experimental-debug-variable-locations=false -stop-before=finalize-isel -o - --try-experimental-debuginfo-iterators | FileCheck %s
; RUN: llc %s -start-after=codegenprepare --experimental-debug-variable-locations -stop-before=finalize-isel -o - --try-experimental-debuginfo-iterators | FileCheck %s

;; Check that unused argument values are handled the same way for variadic
;; dbg_values as non-variadics.

; CHECK: ![[A:[0-9]+]] = !DILocalVariable(name: "a",
; CHECK: ![[B:[0-9]+]] = !DILocalVariable(name: "b",
; CHECK: ![[C:[0-9]+]] = !DILocalVariable(name: "c",

; CHECK: DBG_VALUE $ecx, $noreg, ![[A]], !DIExpression(), debug-location
; CHECK: DBG_VALUE $edx, $noreg, ![[B]], !DIExpression(), debug-location
; CHECK: DBG_VALUE $noreg, $noreg, ![[C]], !DIExpression(), debug-location

target triple = "x86_64-pc-windows-msvc19.16.27034"
define dso_local i32 @"?foo@@YAHHH@Z"(i32 %a, i32 %b) local_unnamed_addr !dbg !8 {
entry:
  call void @llvm.dbg.value(metadata i32 %a, metadata !14, metadata !DIExpression()), !dbg !17
  call void @llvm.dbg.value(metadata i32 %b, metadata !15, metadata !DIExpression()), !dbg !17
  call void @llvm.dbg.value(metadata !DIArgList(i32 %a, i32 %b), metadata !16, metadata !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_arg, 1, DW_OP_plus, DW_OP_stack_value)), !dbg !17
  ret i32 0
}

declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !6}
!llvm.ident = !{!7}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 11.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "test.cpp", directory: "/")
!2 = !{}
!3 = !{i32 2, !"CodeView", i32 1}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 2}
!6 = !{i32 7, !"PIC Level", i32 2}
!7 = !{!"clang version 11.0.0"}
!8 = distinct !DISubprogram(name: "foo", linkageName: "?foo@@YAHHH@Z", scope: !9, file: !9, line: 1, type: !10, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !13)
!9 = !DIFile(filename: "test.cpp", directory: "/")
!10 = !DISubroutineType(types: !11)
!11 = !{!12, !12, !12}
!12 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!13 = !{!14, !15, !16}
!14 = !DILocalVariable(name: "a", arg: 1, scope: !8, file: !9, line: 1, type: !12)
!15 = !DILocalVariable(name: "b", arg: 2, scope: !8, file: !9, line: 1, type: !12)
!16 = !DILocalVariable(name: "c", scope: !8, file: !9, line: 2, type: !12)
!17 = !DILocation(line: 0, scope: !8)
