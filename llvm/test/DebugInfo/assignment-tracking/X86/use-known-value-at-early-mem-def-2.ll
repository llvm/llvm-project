; RUN: llc %s -stop-after=finalize-isel -o - \
; RUN:    -experimental-assignment-tracking \
; RUN: | FileCheck %s --implicit-check-not=DBG_VALUE

;; Check that sandwiching instructions between a linked store and dbg.assign
;; results in a dbg.value(prev_value) being inserted at the store, and a
;; dbg.value(deref) at the dbg.assign.
;; Same as use-known-value-at-early-mem-def.ll except the "early mem def" is
;; for a fragment of the variable rather than the whole.

; CHECK: bb.0.entry:
; CHECK-NEXT: DBG_VALUE %stack.0.c, $noreg, ![[var:[0-9]+]], !DIExpression(DW_OP_deref), debug-location
; CHECK: MOV64mi32 %stack.0.c, 1, $noreg, 0, $noreg, 5
;; No DBG_VALUE required because the stack location is still valid.

; CHECK: MOV32mi %stack.0.c, 1, $noreg, 0, $noreg, 1
; CHECK-NEXT: DBG_VALUE $noreg, $noreg, ![[var]], !DIExpression(DW_OP_LLVM_fragment, 0, 32), debug-location
;; This DBG_VALUE is added by the frag-agg pass because bits [32, 64) are still
;; live in memory.
; CHECK-NEXT: DBG_VALUE %stack.0.c, $noreg, ![[var]], !DIExpression(DW_OP_plus_uconst, 4, DW_OP_deref, DW_OP_LLVM_fragment, 32, 32)

; CHECK: CALL64pcrel32 @d
; CHECK-NEXT: ADJCALLSTACKUP64
; CHECK-NEXT: DBG_VALUE %stack.0.c, $noreg, ![[var]], !DIExpression(DW_OP_deref, DW_OP_LLVM_fragment, 0, 32), debug-location

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nounwind uwtable
define dso_local void @b() local_unnamed_addr #0 !dbg !7 {
entry:
  %c = alloca i64, align 1, !DIAssignID !13
  call void @llvm.dbg.assign(metadata i1 undef, metadata !11, metadata !DIExpression(), metadata !13, metadata i64* %c, metadata !DIExpression()), !dbg !14
  call void @llvm.lifetime.start.p0i64(i64 1, i64* nonnull %c) #4, !dbg !15
  store i64 5, i64* %c, align 1, !dbg !16, !DIAssignID !20
  call void @llvm.dbg.assign(metadata i64 5, metadata !11, metadata !DIExpression(), metadata !20, metadata i64* %c, metadata !DIExpression()), !dbg !14
  tail call void (...) @d() #4, !dbg !21

  ; --- VV  Hand written  VV --- ;
  %bc = bitcast i64* %c to i32*
  store i32 1, i32* %bc, align 1, !dbg !16, !DIAssignID !31
  ;; Check that a dbg.value(undef, frag(0, 32)) is inserted here. The value of
  ;; the fragment is "unknown". TODO: In this case the value of the fragment is
  ;; still obviously 5; a future improvement could be to be smarter and work
  ;; this out. But that's a lot of work for an uncommon case.
  tail call void (...) @d() #4, !dbg !21
  call void @llvm.dbg.assign(metadata i32 1, metadata !11, metadata !DIExpression(DW_OP_LLVM_fragment, 0, 32), metadata !31, metadata i32* %bc, metadata !DIExpression()), !dbg !14
  ; --- AA  Hand written  AA --- ;

  call void @a(i64* nonnull %c) #4, !dbg !22
  call void @llvm.lifetime.end.p0i64(i64 1, i64* nonnull %c) #4, !dbg !23
  ret void, !dbg !23
}

declare void @llvm.lifetime.start.p0i64(i64 immarg, i64* nocapture) #1
declare !dbg !24 dso_local void @d(...) local_unnamed_addr #2
declare !dbg !27 dso_local void @a(i64*) local_unnamed_addr #2
declare void @llvm.lifetime.end.p0i64(i64 immarg, i64* nocapture) #1
declare void @llvm.dbg.assign(metadata, metadata, metadata, metadata, metadata, metadata) #3

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 12.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "reduce.c", directory: "/")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 12.0.0"}
!7 = distinct !DISubprogram(name: "b", scope: !1, file: !1, line: 3, type: !8, scopeLine: 3, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !10)
!8 = !DISubroutineType(types: !9)
!9 = !{null}
!10 = !{!11}
!11 = !DILocalVariable(name: "c", scope: !7, file: !1, line: 4, type: !12)
!12 = !DIBasicType(name: "char", size: 64, encoding: DW_ATE_unsigned)
!13 = distinct !DIAssignID()
!14 = !DILocation(line: 0, scope: !7)
!15 = !DILocation(line: 4, column: 3, scope: !7)
!16 = !DILocation(line: 4, column: 8, scope: !7)
!20 = distinct !DIAssignID()
!21 = !DILocation(line: 5, column: 3, scope: !7)
!22 = !DILocation(line: 6, column: 3, scope: !7)
!23 = !DILocation(line: 7, column: 1, scope: !7)
!24 = !DISubprogram(name: "d", scope: !1, file: !1, line: 2, type: !25, spFlags: DISPFlagOptimized, retainedNodes: !2)
!25 = !DISubroutineType(types: !26)
!26 = !{null, null}
!27 = !DISubprogram(name: "a", scope: !1, file: !1, line: 1, type: !28, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized, retainedNodes: !2)
!28 = !DISubroutineType(types: !29)
!29 = !{null, !30}
!30 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !12, size: 64)
!31 = distinct !DIAssignID()
