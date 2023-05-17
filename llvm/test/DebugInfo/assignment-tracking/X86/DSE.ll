; RUN: llc %s -stop-before=finalize-isel -o - \
; RUN: | FileCheck %s

; Check basic lowering behaviour of dbg.assign intrinsics. The first
; assignment to `local`, which has been DSE'd, should be represented with a
; constant value DBG_VALUE. The second assignment should have a DBG_VALUE
; describing the stack home of the variable.

; $ cat test.c
; void esc(int*);
; void fun() {
;   int local = 5;
;   // ^ killed by v
;   local = 6;
;   esc(&local);
; }
; $ clang -O2 -g -emit -llvm -S test.c -o -

; CHECK: ![[LOCAL:[0-9]+]] = !DILocalVariable(name: "local",
; CHECK: DBG_VALUE 5, $noreg, ![[LOCAL]], !DIExpression(), debug-location ![[DBG:[0-9]+]]
; CHECK-NEXT: MOV32mi [[DEST:.*]], 1, $noreg, 0, $noreg, 6
; CHECK-NEXT: DBG_VALUE [[DEST]], $noreg, ![[LOCAL]], !DIExpression(DW_OP_deref), debug-location ![[DBG]]

target triple = "x86_64-unknown-linux-gnu"

define dso_local void @fun() local_unnamed_addr !dbg !7 {
entry:
  %local = alloca i32, align 4
  call void @llvm.dbg.assign(metadata i32 5, metadata !11, metadata !DIExpression(), metadata !30, metadata ptr %local, metadata !DIExpression()), !dbg !16
  store i32 6, ptr %local, align 4, !dbg !23, !DIAssignID !31
  call void @llvm.dbg.assign(metadata i32 6, metadata !11, metadata !DIExpression(), metadata !31, metadata ptr %local, metadata !DIExpression()), !dbg !16
  call void @esc(ptr nonnull %local), !dbg !24
  ret void, !dbg !25
}

declare void @llvm.dbg.assign(metadata, metadata, metadata, metadata, metadata, metadata)

declare !dbg !26 dso_local void @esc(ptr) local_unnamed_addr

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !1000}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 12.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.c", directory: "/")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 12.0.0"}
!7 = distinct !DISubprogram(name: "fun", scope: !1, file: !1, line: 2, type: !8, scopeLine: 2, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !10)
!8 = !DISubroutineType(types: !9)
!9 = !{null}
!10 = !{!11}
!11 = !DILocalVariable(name: "local", scope: !7, file: !1, line: 3, type: !12)
!12 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!14 = !DIDerivedType(tag: DW_TAG_volatile_type, baseType: !12)
!15 = !DILocation(line: 3, column: 3, scope: !7)
!16 = !DILocation(line: 0, scope: !7)
!17 = !DILocation(line: 4, column: 3, scope: !7)
!18 = !DILocation(line: 4, column: 16, scope: !7)
!23 = !DILocation(line: 5, column: 9, scope: !7)
!24 = !DILocation(line: 6, column: 3, scope: !7)
!25 = !DILocation(line: 7, column: 1, scope: !7)
!26 = !DISubprogram(name: "esc", scope: !1, file: !1, line: 1, type: !27, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized, retainedNodes: !2)
!27 = !DISubroutineType(types: !28)
!28 = !{null, !29}
!29 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !12, size: 64)
!30 = distinct !DIAssignID()
!31 = distinct !DIAssignID()

!1000 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
