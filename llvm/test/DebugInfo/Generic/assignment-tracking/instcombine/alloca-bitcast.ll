; RUN: opt -opaque-pointers=0 -passes=instcombine -S %s -o - -experimental-assignment-tracking \
; RUN: | FileCheck %s

;; NOTE: This test uses typed pointers because it is testing a code path that
;; doesn't get exercised with opaque pointers. If/when PromoteCastOfAllocation
;; is removed from visitBitCast this test should just be deleted.

;; Check that allocas generated in InstCombine's PromoteCastOfAllocation
;; have DIAssignID copied from the original alloca.
;;
;; $ cat reduce.cpp
;; struct c {
;;   c(int);
;;   int a, b;
;; };
;; c d() {
;;   c e(1);
;;   return e;
;; }
;; $ clang -O2 -c -g reduce.cpp -fno-inline -Xclang -disable-llvm-passes -emit-llvm -S \
;;   | opt -opaque-pointers=0 -passes=declare-to-assign -S

; CHECK: entry:
; CHECK-NEXT: %retval = alloca i64, align 8, !DIAssignID ![[ID:[0-9]+]]
; CHECK-NEXT: %tmpcast = bitcast i64* %retval to %struct.c*
; CHECK-NEXT: call void @llvm.dbg.assign(metadata i1 undef, metadata ![[e:[0-9]+]], metadata !DIExpression(), metadata ![[ID]], metadata i64* %retval, metadata !DIExpression()), !dbg
; CHECK: ![[e]] = !DILocalVariable(name: "e",

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

%struct.c = type { i32, i32 }

define dso_local i64 @_Z1dv() !dbg !7 {
entry:
  %retval = alloca %struct.c, align 4, !DIAssignID !21
  call void @llvm.dbg.assign(metadata i1 undef, metadata !20, metadata !DIExpression(), metadata !21, metadata %struct.c* %retval, metadata !DIExpression()), !dbg !22
  call void @_ZN1cC1Ei(%struct.c* %retval, i32 1), !dbg !23
  %0 = bitcast %struct.c* %retval to i64*, !dbg !24
  %1 = load i64, i64* %0, align 4, !dbg !24
  ret i64 %1, !dbg !24
}

declare dso_local void @_ZN1cC1Ei(%struct.c*, i32) unnamed_addr
declare void @llvm.dbg.assign(metadata, metadata, metadata, metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 12.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "reduce.cpp", directory: "/")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 12.0.0"}
!7 = distinct !DISubprogram(name: "d", linkageName: "_Z1dv", scope: !1, file: !1, line: 5, type: !8, scopeLine: 5, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !19)
!8 = !DISubroutineType(types: !9)
!9 = !{!10}
!10 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "c", file: !1, line: 1, size: 64, flags: DIFlagTypePassByValue | DIFlagNonTrivial, elements: !11, identifier: "_ZTS1c")
!11 = !{!12, !14, !15}
!12 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !10, file: !1, line: 3, baseType: !13, size: 32)
!13 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!14 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !10, file: !1, line: 3, baseType: !13, size: 32, offset: 32)
!15 = !DISubprogram(name: "c", scope: !10, file: !1, line: 2, type: !16, scopeLine: 2, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!16 = !DISubroutineType(types: !17)
!17 = !{null, !18, !13}
!18 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !10, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!19 = !{!20}
!20 = !DILocalVariable(name: "e", scope: !7, file: !1, line: 6, type: !10)
!21 = distinct !DIAssignID()
!22 = !DILocation(line: 0, scope: !7)
!23 = !DILocation(line: 6, column: 5, scope: !7)
!24 = !DILocation(line: 7, column: 3, scope: !7)
