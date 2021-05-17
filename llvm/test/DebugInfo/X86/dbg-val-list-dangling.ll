;; At the moment we emit an undef as soon as we encounter "dangling" variadic
;; dbg_value nodes. This does not reduce correctness but does reduce coverage.
;; We should make variadic dbg_values work in the same way as their
;; non-variadic counterparts here.
;; FIXME: When dangling nodes for a variadic dbg_value are found, we should be
;; able to recover the value in some cases.

; RUN: llc %s -start-after=codegenprepare -stop-before=finalize-isel -o - | FileCheck %s

;; Check that dangling debug info in the SelectionDAG build phase is handled
;; in the same way for variadic dbg_value ndoes as non-variadics.

;; Generated from the following source with -g -O2. Second dbg.value modified
;; to be variadic by hand:
;; void a(char*);
;; void b() {
;;   char *c = "abc";
;;   char *d = "abc";
;;   a("abc");
;; }

; CHECK: ![[C:[0-9]+]] = !DILocalVariable(name: "c",
; CHECK: ![[D:[0-9]+]] = !DILocalVariable(name: "d",

; CHECK-DAG: DBG_VALUE %[[VREG:[0-9]]], $noreg, ![[C]], !DIExpression(), debug-location
; CHECK-DAG: DBG_VALUE_LIST ![[D]], !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_stack_value), $noreg, debug-location

target triple = "x86_64-unknown-linux-gnu"

@.str = private unnamed_addr constant [4 x i8] c"abc\00", align 1

define dso_local void @b() local_unnamed_addr !dbg !7 {
entry:
  call void @llvm.dbg.value(metadata i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str, i64 0, i64 0), metadata !11, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.value(metadata !DIArgList(i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str, i64 0, i64 0)), metadata !14, metadata !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_stack_value)), !dbg !15
  tail call void @a(i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str, i64 0, i64 0)) #3, !dbg !16
  ret void, !dbg !17
}

declare !dbg !18 dso_local void @a(i8*) local_unnamed_addr
declare void @llvm.dbg.value(metadata, metadata, metadata)


!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 11.0.0)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.c", directory: "/")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 11.0.0"}
!7 = distinct !DISubprogram(name: "b", scope: !1, file: !1, line: 2, type: !8, scopeLine: 2, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !10)
!8 = !DISubroutineType(types: !9)
!9 = !{null}
!10 = !{!11, !14}
!11 = !DILocalVariable(name: "c", scope: !7, file: !1, line: 3, type: !12)
!12 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !13, size: 64)
!13 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!14 = !DILocalVariable(name: "d", scope: !7, file: !1, line: 4, type: !12)
!15 = !DILocation(line: 0, scope: !7)
!16 = !DILocation(line: 5, column: 3, scope: !7)
!17 = !DILocation(line: 6, column: 1, scope: !7)
!18 = !DISubprogram(name: "a", scope: !1, file: !1, line: 1, type: !19, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized, retainedNodes: !2)
!19 = !DISubroutineType(types: !20)
!20 = !{null, !12}
