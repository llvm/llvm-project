; RUN: opt -passes=sroa,verify -S %s -o - \
; RUN: | FileCheck %s --implicit-check-not="call void @llvm.dbg"
; RUN: opt --try-experimental-debuginfo-iterators -passes=sroa,verify -S %s -o - \
; RUN: | FileCheck %s --implicit-check-not="call void @llvm.dbg"

; Check that single sliced allocas retain their assignment tracking debug info.

;; $ cat test.c
;; struct a {
;;   char b[8];
;; };
;; int c;
;; void d() {
;;   struct a a;
;;   memcpy(a.b, 0, c);
;; }
;; $ clang test.c -Xclang -disable-llvm-passes -O2 -g -c -S -emit-llvm -o - \
;;   | opt -passes=declare-to-assign -S -o -

; CHECK: entry:
; CHECK-NEXT: %a.sroa.0 = alloca i64, align 8, !DIAssignID ![[ID_1:[0-9]+]]
; CHECK-NEXT: call void @llvm.dbg.assign(metadata i1 undef, metadata ![[VAR:[0-9]+]], metadata !DIExpression(), metadata ![[ID_1]], metadata ptr %a.sroa.0, metadata !DIExpression()), !dbg

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

%struct.a = type { [8 x i8] }

@c = dso_local global i32 0, align 4, !dbg !0

define dso_local void @d() !dbg !11 {
entry:
  %a = alloca %struct.a, align 1, !DIAssignID !23
  call void @llvm.dbg.assign(metadata i1 undef, metadata !15, metadata !DIExpression(), metadata !23, metadata ptr %a, metadata !DIExpression()), !dbg !24
  %0 = bitcast ptr %a to ptr, !dbg !25
  call void @llvm.lifetime.start.p0i8(i64 8, ptr %0), !dbg !25
  %b = getelementptr inbounds %struct.a, ptr %a, i32 0, i32 0, !dbg !26
  %arraydecay = getelementptr inbounds [8 x i8], ptr %b, i64 0, i64 0, !dbg !27
  %1 = load i32, ptr @c, align 4, !dbg !28
  %conv = sext i32 %1 to i64, !dbg !28
  call void @llvm.memcpy.p0i8.p0i8.i64(ptr align 1 %arraydecay, ptr align 1 null, i64 %conv, i1 false), !dbg !27
  %2 = bitcast ptr %a to ptr, !dbg !33
  call void @llvm.lifetime.end.p0i8(i64 8, ptr %2), !dbg !33
  ret void, !dbg !33
}

declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture)
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* noalias nocapture writeonly, i8* noalias nocapture readonly, i64, i1 immarg)
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture)
declare void @llvm.dbg.assign(metadata, metadata, metadata, metadata, metadata, metadata)

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!7, !8, !9, !1000}
!llvm.ident = !{!10}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "c", scope: !2, file: !3, line: 4, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 12.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "test.c", directory: "/")
!4 = !{}
!5 = !{!0}
!6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!7 = !{i32 7, !"Dwarf Version", i32 4}
!8 = !{i32 2, !"Debug Info Version", i32 3}
!9 = !{i32 1, !"wchar_size", i32 4}
!10 = !{!"clang version 12.0.0"}
!11 = distinct !DISubprogram(name: "d", scope: !3, file: !3, line: 5, type: !12, scopeLine: 5, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !14)
!12 = !DISubroutineType(types: !13)
!13 = !{null}
!14 = !{!15}
!15 = !DILocalVariable(name: "a", scope: !11, file: !3, line: 6, type: !16)
!16 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "a", file: !3, line: 1, size: 64, elements: !17)
!17 = !{!18}
!18 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !16, file: !3, line: 2, baseType: !19, size: 64)
!19 = !DICompositeType(tag: DW_TAG_array_type, baseType: !20, size: 64, elements: !21)
!20 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!21 = !{!22}
!22 = !DISubrange(count: 8)
!23 = distinct !DIAssignID()
!24 = !DILocation(line: 0, scope: !11)
!25 = !DILocation(line: 6, column: 3, scope: !11)
!26 = !DILocation(line: 7, column: 12, scope: !11)
!27 = !DILocation(line: 7, column: 3, scope: !11)
!28 = !DILocation(line: 7, column: 18, scope: !11)
!33 = !DILocation(line: 8, column: 1, scope: !11)
!1000 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
