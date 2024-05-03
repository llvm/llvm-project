; RUN: opt -S -passes=instcombine %s | FileCheck %s -check-prefix=RUN-ONCE
; RUN: opt -S -passes=instcombine %s --try-experimental-debuginfo-iterators | FileCheck %s -check-prefix=RUN-ONCE

; This example was reduced from a test case in which InstCombine ran at least
; twice:
;   - The first InstCombine run converted dbg.declares to dbg.values using the
;     LowerDbgDeclare utility. This produced a dbg.value(ptr %2, DW_OP_deref)
;     (this happens when the contents of an alloca are passed by-value), and a
;     dbg.value(i32 %0) (due to the store of %0 into the alloca).
;   - The second InstCombine run deleted the alloca (%2).
; Check that the DW_OP_deref dbg.value is deleted, just like a dbg.declare would
; be.
;
; RUN-ONCE-LABEL: @t1(
; RUN-ONCE-NEXT: llvm.dbg.value(metadata i32 %0, metadata [[t1_arg0:![0-9]+]], metadata !DIExpression())
; RUN-ONCE-NEXT: llvm.dbg.value(metadata ptr poison, metadata [[t1_fake_ptr:![0-9]+]], metadata !DIExpression())
; RUN-ONCE-NEXT: ret void
define void @t1(i32) !dbg !9 {
  %2 = alloca i32, align 4
  store i32 %0, ptr %2, align 4
  call void @llvm.dbg.value(metadata i32 %0, metadata !14, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.value(metadata ptr %2, metadata !14, metadata !DIExpression(DW_OP_deref)), !dbg !15
  call void @llvm.dbg.value(metadata ptr %2, metadata !20, metadata !DIExpression()), !dbg !15
  ret void
}

; This example is closer to an end-to-end test: the IR looks like it could have
; been produced by a frontend compiling at -O0.
;
; Here's what happens:
; 1) We run InstCombine. This puts a dbg.value(ptr %x.addr, DW_OP_deref)
;    before the call to @use, and a dbg.value(i32 %x) after the store.
; 2) We inline @use.
; 3) We run InstCombine again. The alloca %x.addr is erased. We should just get
;    dbg.value(i32 %x). There should be no leftover dbg.value(metadata ptr
;    undef).
;
;;; define void @use(ptr %addr) alwaysinline { ret void }
;;; define void @t2(i32 %x) !dbg !17 {
;;;   %x.addr = alloca i32, align 4
;;;   store i32 %x, ptr %x.addr, align 4
;;;   call void @llvm.dbg.declare(metadata ptr %x.addr, metadata !18, metadata !DIExpression()), !dbg !19
;;;   call void @use(ptr %x.addr)
;;;   ret void
;;; }

declare void @llvm.dbg.value(metadata, metadata, metadata)
declare void @llvm.dbg.declare(metadata, metadata, metadata)

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.dbg.cu = !{!5}
!llvm.ident = !{!8}

; RUN-ONCE: [[t1_arg0]] = !DILocalVariable(name: "a"
; RUN-ONCE: [[t1_fake_ptr]] = !DILocalVariable(name: "fake_ptr"

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 10, i32 14]}
!1 = !{i32 2, !"Dwarf Version", i32 4}
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !{i32 1, !"wchar_size", i32 4}
!4 = !{i32 7, !"PIC Level", i32 2}
!5 = distinct !DICompileUnit(language: DW_LANG_C99, file: !6, producer: "", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !7, nameTableKind: GNU)
!6 = !DIFile(filename: "-", directory: "/")
!7 = !{}
!8 = !{!""}
!9 = distinct !DISubprogram(name: "t1", scope: !10, file: !10, line: 1, type: !11, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !7)
!10 = !DIFile(filename: "<stdin>", directory: "/")
!11 = !DISubroutineType(types: !12)
!12 = !{null, !13}
!13 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!14 = !DILocalVariable(name: "a", arg: 1, scope: !9, file: !10, line: 1, type: !13)
!15 = !DILocation(line: 1, column: 13, scope: !9)
!16 = !DILocation(line: 1, column: 17, scope: !9)
!17 = distinct !DISubprogram(name: "t2", scope: !10, file: !10, line: 1, type: !11, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !7)
!18 = !DILocalVariable(name: "x", arg: 1, scope: !17, file: !10, line: 1, type: !13)
!19 = !DILocation(line: 1, column: 1, scope: !17)
!20 = !DILocalVariable(name: "fake_ptr", scope: !9, file: !10, line: 1, type: !13)
