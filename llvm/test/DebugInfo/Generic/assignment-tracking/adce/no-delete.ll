; RUN: opt %s -passes=adce -S -o - \
; RUN: | FileCheck %s

;; $ cat test.c
;; void fun(int local) {}
;;
;; $ clang test.c -g -O1 -Xclang -disable-llvm-passes | opt -S -passes=declare-to-assign

;; Check that the dbg.assign intrinsics that are considered "out of scope" (dbg
;; attachments have been deleted) but still linked to an instruction are not
;; deleted by ADCE.

; CHECK: llvm.dbg.assign
; CHECK: llvm.dbg.assign

define dso_local void @fun(i32 noundef %local) #0 !dbg !7 {
entry:
  %local.addr = alloca i32, align 4, !DIAssignID !13
  call void @llvm.dbg.assign(metadata i1 undef, metadata !12, metadata !DIExpression(), metadata !13, metadata ptr %local.addr, metadata !DIExpression()), !dbg !14
  store i32 %local, ptr %local.addr, align 4, !DIAssignID !19
  call void @llvm.dbg.assign(metadata i32 %local, metadata !12, metadata !DIExpression(), metadata !19, metadata ptr %local.addr, metadata !DIExpression()), !dbg !14
  ret void
}

declare void @llvm.dbg.declare(metadata, metadata, metadata)
declare void @llvm.dbg.assign(metadata, metadata, metadata, metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !1000}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 14.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.c", directory: "/")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 7, !"uwtable", i32 1}
!6 = !{!"clang version 14.0.0"}
!7 = distinct !DISubprogram(name: "fun", scope: !1, file: !1, line: 1, type: !8, scopeLine: 1, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !11)
!8 = !DISubroutineType(types: !9)
!9 = !{null, !10}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !{!12}
!12 = !DILocalVariable(name: "local", arg: 1, scope: !7, file: !1, line: 1, type: !10)
!13 = distinct !DIAssignID()
!14 = !DILocation(line: 0, scope: !7)
!19 = distinct !DIAssignID()
!20 = !DILocation(line: 1, column: 22, scope: !7)
!1000 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
