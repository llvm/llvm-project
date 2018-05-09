; RUN: opt -sroa -S -o - %s | FileCheck %s
; Generated from clang -c  -O2 -g -target x86_64-pc-windows-msvc
; struct A {
;   int _Myval2;
;   A() : _Myval2() {}
; };
; struct B {
;   double buffer[];
; };
; struct C {
;   C(int) {}
;   A _Mypair;
; };
; int getPtr();
; struct D {
;   C takePayload() {
;     C Tmp(getPtr());
;     return Tmp;
;   }
; } Dd;
; void *operator new(size_t, void *);
; struct F {
;   F(D Err) : HasError() {
;     C *e = (C *)(ErrorStorage.buffer);
;     new (e) C(Err.takePayload());
;   }
;   B ErrorStorage;
;   bool HasError;
; };
; F fn2() { return Dd; }
; void fn3() { fn2(); }
source_filename = "test.ll"

%struct.F = type { %struct.B, i8 }
%struct.B = type { [0 x double], [8 x i8] }

define void @"\01?fn3@@YAXXZ"() local_unnamed_addr !dbg !6 {
entry:
  %tmp = alloca %struct.F, align 8
  %0 = bitcast %struct.F* %tmp to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* %0)
  call void @llvm.dbg.declare(metadata %struct.F* %tmp, metadata !10, metadata !DIExpression()), !dbg !14
  ; CHECK-NOT: !DIExpression(DW_OP_LLVM_fragment, 32, 96)
  ; CHECK: call void @llvm.dbg.value(metadata i32 0, metadata !10, metadata !DIExpression())
  %_Myval2.i.i.i.i.i = bitcast %struct.F* %tmp to i32*
  store i32 0, i32* %_Myval2.i.i.i.i.i, align 8
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #0

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

attributes #0 = { argmemonly nounwind }
attributes #1 = { nounwind readnone speculatable }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 6.0.0 (trunk 319178) (llvm/trunk 319187)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "test.cpp", directory: "/")
!2 = !{i32 2, !"Dwarf Version", i32 4}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 2}
!5 = !{i32 7, !"PIC Level", i32 2}
!6 = distinct !DISubprogram(name: "fn3", linkageName: "\01?fn3@@YAXXZ", scope: !1, file: !1, line: 30, type: !7, isLocal: false, isDefinition: true, scopeLine: 30, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !9)
!7 = !DISubroutineType(types: !8)
!8 = !{null}
!9 = !{}
!10 = !DILocalVariable(name: "Tmp", scope: !11, file: !1, line: 16, type: !23)
!11 = distinct !DISubprogram(name: "takePayload", linkageName: "\01?takePayload@D@@QEAA?AUC@@XZ", scope: !12, file: !1, line: 15, type: !7, isLocal: false, isDefinition: true, scopeLine: 15, flags: DIFlagPrototyped, isOptimized: true, unit: !0, declaration: !13, retainedNodes: !9)
!12 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "D", file: !1, line: 14, size: 8, elements: !9, identifier: ".?AUD@@")
!13 = !DISubprogram(name: "takePayload", linkageName: "\01?takePayload@D@@QEAA?AUC@@XZ", scope: !12, file: !1, line: 15, type: !7, isLocal: false, isDefinition: false, scopeLine: 15, flags: DIFlagPrototyped, isOptimized: true)
!14 = !DILocation(line: 16, column: 7, scope: !11, inlinedAt: !15)
!15 = distinct !DILocation(line: 24, column: 19, scope: !16, inlinedAt: !20)
!16 = distinct !DILexicalBlock(scope: !17, file: !1, line: 22, column: 25)
!17 = distinct !DISubprogram(name: "F", linkageName: "\01??0F@@QEAA@UD@@@Z", scope: !18, file: !1, line: 22, type: !7, isLocal: false, isDefinition: true, scopeLine: 22, flags: DIFlagPrototyped, isOptimized: true, unit: !0, declaration: !19, retainedNodes: !9)
!18 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "F", file: !1, line: 21, size: 128, elements: !9, identifier: ".?AUF@@")
!19 = !DISubprogram(name: "F", scope: !18, file: !1, line: 22, type: !7, isLocal: false, isDefinition: false, scopeLine: 22, flags: DIFlagPrototyped, isOptimized: true)
!20 = distinct !DILocation(line: 29, column: 18, scope: !21, inlinedAt: !22)
!21 = distinct !DISubprogram(name: "fn2", linkageName: "\01?fn2@@YA?AUF@@XZ", scope: !1, file: !1, line: 29, type: !7, isLocal: false, isDefinition: true, scopeLine: 29, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !9)
!22 = distinct !DILocation(line: 30, column: 14, scope: !6)
!23 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "C", file: !1, line: 9, size: 32, elements: !9, identifier: ".?AUC@@")
