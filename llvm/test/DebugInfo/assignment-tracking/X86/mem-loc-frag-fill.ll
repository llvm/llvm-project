; RUN: llc %s -stop-before finalize-isel -o - \
; RUN:    -experimental-debug-variable-locations=false \
; RUN: | FileCheck %s --implicit-check-not=DBG_
; RUN: llc %s -stop-before finalize-isel -o - \
; RUN:    -experimental-debug-variable-locations=true \
; RUN: | FileCheck %s  --implicit-check-not=DBG_

;; Check that the mem-loc-frag-fill analysis works on a simple case; ensure
;; that location definitions are added to preserve memory locations of
;; fragments of variables at subsequent location definitions for other
;; fragments of the variable are not currently in memory.

;; Test generated from:
;; $ cat test.cpp
;; struct Nums { int a, b, c; };
;; void esc(struct Nums*);
;; void step();
;; int main() {
;;   struct Nums nums = { 1, 2, 1 }; //< Store to .c is elided.
;;   step();
;;   nums.c = 2; //< Killing store.
;;   step();
;;   esc(&nums);
;;   return 0;
;; }
;; $ clang++ test.cpp -O2 -g -Xclang -fexperimental-assignmment-tracking -emit-llvm -S -o -

;; Most check lines are inline in main.
; CHECK: ![[nums:[0-9]+]] = !DILocalVariable(name: "nums",

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.Nums = type { i32, i32, i32 }

; Function Attrs: mustprogress norecurse uwtable
define dso_local noundef i32 @main() local_unnamed_addr #0 !dbg !7 {
entry:
  %nums = alloca %struct.Nums, align 8, !DIAssignID !18
; CHECK: DBG_VALUE %stack.0.nums, $noreg, ![[nums]], !DIExpression(DW_OP_deref)
  call void @llvm.dbg.assign(metadata i1 undef, metadata !12, metadata !DIExpression(), metadata !18, metadata ptr %nums, metadata !DIExpression()), !dbg !19
  store i64 8589934593, ptr %nums, align 8, !dbg !21, !DIAssignID !22
; CHECK: MOV64mr %stack.0.nums, 1, $noreg, 0, $noreg, killed %0
  call void @llvm.dbg.assign(metadata i1 undef, metadata !12, metadata !DIExpression(), metadata !22, metadata ptr %nums, metadata !DIExpression()), !dbg !19
  call void @llvm.dbg.assign(metadata i1 undef, metadata !12, metadata !DIExpression(DW_OP_LLVM_fragment, 64, 32), metadata !22, metadata ptr undef, metadata !DIExpression()), !dbg !19
; CHECK-NEXT: DBG_VALUE $noreg, $noreg, ![[nums]], !DIExpression(DW_OP_LLVM_fragment, 64, 32)
; CHECK-NEXT: DBG_VALUE %stack.0.nums, $noreg, ![[nums]], !DIExpression(DW_OP_deref, DW_OP_LLVM_fragment, 0, 64)
  tail call void @_Z4stepv(), !dbg !23
  %c = getelementptr inbounds %struct.Nums, ptr %nums, i64 0, i32 2, !dbg !24
  store i32 2, ptr %c, align 8, !dbg !25, !DIAssignID !31
; CHECK: MOV32mi %stack.0.nums, 1, $noreg, 8, $noreg, 2
  call void @llvm.dbg.assign(metadata i32 2, metadata !12, metadata !DIExpression(DW_OP_LLVM_fragment, 64, 32), metadata !31, metadata ptr %c, metadata !DIExpression()), !dbg !19
; CHECK-NEXT: DBG_VALUE %stack.0.nums, $noreg, ![[nums]], !DIExpression(DW_OP_plus_uconst, 8, DW_OP_deref, DW_OP_LLVM_fragment, 64, 32)
  tail call void @_Z4stepv(), !dbg !32
  call void @_Z3escP4Nums(ptr noundef nonnull %nums), !dbg !33
  ret i32 0, !dbg !35
}

declare void @llvm.lifetime.start.p0i8(i64 immarg, ptr nocapture) #1
declare !dbg !36 dso_local void @_Z4stepv() local_unnamed_addr #2
declare !dbg !40 dso_local void @_Z3escP4Nums(ptr noundef) local_unnamed_addr #2
declare void @llvm.lifetime.end.p0i8(i64 immarg, ptr nocapture) #1
declare void @llvm.dbg.assign(metadata, metadata, metadata, metadata, metadata, metadata) #3

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !1000}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 14.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.cpp", directory: "/")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 7, !"uwtable", i32 1}
!6 = !{!"clang version 14.0.0"}
!7 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 4, type: !8, scopeLine: 4, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !11)
!8 = !DISubroutineType(types: !9)
!9 = !{!10}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !{!12}
!12 = !DILocalVariable(name: "nums", scope: !7, file: !1, line: 5, type: !13)
!13 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Nums", file: !1, line: 1, size: 96, flags: DIFlagTypePassByValue, elements: !14, identifier: "_ZTS4Nums")
!14 = !{!15, !16, !17}
!15 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !13, file: !1, line: 1, baseType: !10, size: 32)
!16 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !13, file: !1, line: 1, baseType: !10, size: 32, offset: 32)
!17 = !DIDerivedType(tag: DW_TAG_member, name: "c", scope: !13, file: !1, line: 1, baseType: !10, size: 32, offset: 64)
!18 = distinct !DIAssignID()
!19 = !DILocation(line: 0, scope: !7)
!20 = !DILocation(line: 5, column: 3, scope: !7)
!21 = !DILocation(line: 5, column: 15, scope: !7)
!22 = distinct !DIAssignID()
!23 = !DILocation(line: 6, column: 3, scope: !7)
!24 = !DILocation(line: 7, column: 8, scope: !7)
!25 = !DILocation(line: 7, column: 10, scope: !7)
!31 = distinct !DIAssignID()
!32 = !DILocation(line: 8, column: 3, scope: !7)
!33 = !DILocation(line: 9, column: 3, scope: !7)
!34 = !DILocation(line: 11, column: 1, scope: !7)
!35 = !DILocation(line: 10, column: 3, scope: !7)
!36 = !DISubprogram(name: "step", linkageName: "_Z4stepv", scope: !1, file: !1, line: 3, type: !37, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized, retainedNodes: !39)
!37 = !DISubroutineType(types: !38)
!38 = !{null}
!39 = !{}
!40 = !DISubprogram(name: "esc", linkageName: "_Z3escP4Nums", scope: !1, file: !1, line: 2, type: !41, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized, retainedNodes: !39)
!41 = !DISubroutineType(types: !42)
!42 = !{null, !43}
!43 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !13, size: 64)
!1000 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
