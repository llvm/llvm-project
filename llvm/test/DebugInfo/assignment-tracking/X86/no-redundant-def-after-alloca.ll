; RUN: llc %s -o - -stop-after=finalize-isel \
; RUN:    -experimental-assignment-tracking  \
; RUN: | FileCheck %s --implicit-check-not=DBG_

;; Hand written. Check that no unnecessary undef is inserted after an alloca
;; that has a linked dbg.assign that doesn't immediately follow it.

; CHECK: CALL64pcrel32 @a
; CHECK-NEXT: ADJCALLSTACKUP64
; CHECK-NEXT: DBG_VALUE %stack.0.c, $noreg, !{{.+}}, !DIExpression(DW_OP_deref), debug-location

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define dso_local void @b() #0 !dbg !7 {
entry:
  %c = alloca i8, align 1, !DIAssignID !10
  call void (...) @a(), !dbg !16
  call void @llvm.dbg.assign(metadata i1 undef, metadata !11, metadata !DIExpression(), metadata !10, metadata ptr %c, metadata !DIExpression()), !dbg !13
  ret void, !dbg !17
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) #1
declare dso_local void @a(...)
declare void @llvm.dbg.assign(metadata, metadata, metadata, metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 12.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.c", directory: "/")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 12.0.0)"}
!7 = distinct !DISubprogram(name: "b", scope: !1, file: !1, line: 2, type: !8, scopeLine: 2, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!8 = !DISubroutineType(types: !9)
!9 = !{null}
!10 = distinct !DIAssignID()
!11 = !DILocalVariable(name: "c", scope: !7, file: !1, line: 3, type: !12)
!12 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!13 = !DILocation(line: 0, scope: !7)
!14 = !DILocation(line: 3, column: 8, scope: !7)
!15 = distinct !DIAssignID()
!16 = !DILocation(line: 4, column: 3, scope: !7)
!17 = !DILocation(line: 5, column: 1, scope: !7)
