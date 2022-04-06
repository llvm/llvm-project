; RUN: llc %s -o - -stop-after=livedebugvalues -opt-bisect-limit=0 \
; RUN:   | FileCheck %s
; RUN: llc %s -o - -stop-after=livedebugvalues -opt-bisect-limit=10 \
; RUN:   | FileCheck %s
; RUN: llc %s -o - -stop-after=livedebugvalues -opt-bisect-limit=20 \
; RUN:   | FileCheck %s
; RUN: llc %s -o - -stop-after=livedebugvalues -opt-bisect-limit=30 \
; RUN:   | FileCheck %s
; RUN: llc %s -o - -stop-after=livedebugvalues -opt-bisect-limit=40 \
; RUN:   | FileCheck %s
; RUN: llc %s -o - -stop-after=livedebugvalues -opt-bisect-limit=100 \
; RUN:   | FileCheck %s
;; Test fast-isel for good measure too,
; RUN: llc %s -o - -stop-after=livedebugvalues -fast-isel=true \
; RUN:   | FileCheck %s
; RUN: llc %s -o - -stop-after=livedebugvalues -fast-isel=true \
; RUN:   -opt-bisect-limit=0 | FileCheck %s
; RUN: llc %s -o - -stop-after=livedebugvalues -fast-isel=true \
; RUN:   -opt-bisect-limit=10 | FileCheck %s
; RUN: llc %s -o - -stop-after=livedebugvalues -fast-isel=true \
; RUN:   -opt-bisect-limit=100 | FileCheck %s

; The function below should be optimised with the "default" optimisation level.
; However, opt-bisect-limit causes SelectionDAG to change the target settings
; for the duration of SelectionDAG. This can lead to variable locations created
; in one mode, but the rest of the debug-info analyses expecting the other.
; Test that opt-bisect-limit can be used without any assertions firing, and
; that instruction referencing instructions are seen each time.
;
; The selection of bisect positions above are not picked meaningfully, the
; pass order and positioning will change over time. The most important part
; is that the bisect limit sits between SelectionDAG and debug-info passes at
; some point.

; CHECK: DBG_INSTR_REF
; CHECK: DBG_PHI

; ModuleID = '/tmp/test.cpp'
source_filename = "/tmp/test.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%class.Class = type { i8 }

; Function Attrs: mustprogress uwtable
define dso_local void @_Z4FuncP5Class(%class.Class* noundef %c) local_unnamed_addr !dbg !7 {
entry:
  call void @llvm.dbg.value(metadata %class.Class* %c, metadata !15, metadata !DIExpression()), !dbg !16
  %tobool.not = icmp eq %class.Class* %c, null, !dbg !17
  br i1 %tobool.not, label %land.lhs.true, label %if.end, !dbg !19

land.lhs.true:                                    ; preds = %entry
  %call = call noundef zeroext i1 @_Z4Condv(), !dbg !20
  call void @llvm.assume(i1 %call), !dbg !21
  %call1 = call noalias noundef nonnull dereferenceable(1) i8* @_Znwm(i64 noundef 1), !dbg !22, !heapallocsite !12
  %0 = bitcast i8* %call1 to %class.Class*, !dbg !22
  call void @llvm.dbg.value(metadata %class.Class* %0, metadata !15, metadata !DIExpression()), !dbg !16
  br label %if.end, !dbg !23

if.end:                                           ; preds = %land.lhs.true, %entry
  %c.addr.0 = phi %class.Class* [ %c, %entry ], [ %0, %land.lhs.true ]
  call void @llvm.dbg.value(metadata %class.Class* %c.addr.0, metadata !15, metadata !DIExpression()), !dbg !16
  call void @_Z11DoSomethingR5Class(%class.Class* noundef nonnull align 1 dereferenceable(1) %c.addr.0), !dbg !24
  ret void, !dbg !25
}

declare !dbg !26 dso_local noundef zeroext i1 @_Z4Condv() local_unnamed_addr

; Function Attrs: nobuiltin allocsize(0)
declare dso_local noundef nonnull i8* @_Znwm(i64 noundef) local_unnamed_addr

declare !dbg !30 dso_local void @_Z11DoSomethingR5Class(%class.Class* noundef nonnull align 1 dereferenceable(1)) local_unnamed_addr

; Function Attrs: nocallback nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.value(metadata, metadata, metadata)

; Function Attrs: inaccessiblememonly nocallback nofree nosync nounwind willreturn
declare void @llvm.assume(i1 noundef)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "/tmp/test.cpp", directory: ".")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 7, !"uwtable", i32 2}
!6 = !{!"clang"}
!7 = distinct !DISubprogram(name: "Func", linkageName: "_Z4FuncP5Class", scope: !8, file: !8, line: 6, type: !9, scopeLine: 6, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !14)
!8 = !DIFile(filename: "/tmp/test.cpp", directory: "")
!9 = !DISubroutineType(types: !10)
!10 = !{null, !11}
!11 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !12, size: 64)
!12 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "Class", file: !8, line: 3, size: 8, flags: DIFlagTypePassByValue, elements: !13, identifier: "_ZTS5Class")
!13 = !{}
!14 = !{!15}
!15 = !DILocalVariable(name: "c", arg: 1, scope: !7, file: !8, line: 6, type: !11)
!16 = !DILocation(line: 0, scope: !7)
!17 = !DILocation(line: 7, column: 10, scope: !18)
!18 = distinct !DILexicalBlock(scope: !7, file: !8, line: 7, column: 9)
!19 = !DILocation(line: 7, column: 12, scope: !18)
!20 = !DILocation(line: 7, column: 15, scope: !18)
!21 = !DILocation(line: 7, column: 9, scope: !7)
!22 = !DILocation(line: 8, column: 13, scope: !18)
!23 = !DILocation(line: 8, column: 9, scope: !18)
!24 = !DILocation(line: 10, column: 5, scope: !7)
!25 = !DILocation(line: 11, column: 1, scope: !7)
!26 = !DISubprogram(name: "Cond", linkageName: "_Z4Condv", scope: !8, file: !8, line: 5, type: !27, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized, retainedNodes: !13)
!27 = !DISubroutineType(types: !28)
!28 = !{!29}
!29 = !DIBasicType(name: "bool", size: 8, encoding: DW_ATE_boolean)
!30 = !DISubprogram(name: "DoSomething", linkageName: "_Z11DoSomethingR5Class", scope: !8, file: !8, line: 4, type: !31, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized, retainedNodes: !13)
!31 = !DISubroutineType(types: !32)
!32 = !{null, !33}
!33 = !DIDerivedType(tag: DW_TAG_reference_type, baseType: !12, size: 64)
