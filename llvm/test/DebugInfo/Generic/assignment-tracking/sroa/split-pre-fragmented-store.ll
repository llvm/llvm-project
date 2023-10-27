; RUN: opt -S -passes=sroa -sroa-skip-mem2reg %s \
; RUN: | FileCheck %s --implicit-check-not="call void @llvm.dbg"

;; IR hand-modified, originally generated from:
;; struct Pair { int a; int b; };
;; Pair getVar();
;; int fun() {
;;   Pair var;
;;   var = getVar();
;;   return var.b;
;; }
;; Modification: split the dbg.assign linked the the memcpy(64 bits) into two,
;; each describing a 32 bit fragment.
;;
;; Check that assignment tracking updates in SROA work when the store being
;; split is described with one dbg.assign (covering different fragments). The
;; store may have been already split and then merged again at some point.

;; Alloca for var.a and associated dbg.assign:
; CHECK: %var.sroa.0 = alloca i32, align 4, !DIAssignID ![[id_1:[0-9]+]]
; CHECK-NEXT: call void @llvm.dbg.assign(metadata i1 undef, metadata ![[var:[0-9]+]], metadata !DIExpression(DW_OP_LLVM_fragment, 0, 32), metadata ![[id_1]], metadata ptr %var.sroa.0, metadata !DIExpression())

;; Alloca for var.b and associated dbg.assign:
; CHECK-NEXT: %var.sroa.1 = alloca i32, align 4, !DIAssignID ![[id_2:[0-9]+]]
; CHECK-NEXT: call void @llvm.dbg.assign(metadata i1 undef, metadata ![[var]], metadata !DIExpression(DW_OP_LLVM_fragment, 32, 32), metadata ![[id_2]], metadata ptr %var.sroa.1, metadata !DIExpression())

;; Store to var.b (split from store to var) and associated dbg.assigns. The
;; dbg.assign for the fragment covering the (pre-split) assignment to var.a
;; should not be linked to the store.
; CHECK: store i32 %[[v:.*]], ptr %var.sroa.1,{{.*}}!DIAssignID ![[id_3:[0-9]+]]
; CHECK-NEXT: call void @llvm.dbg.assign(metadata i32 %{{.*var\.sroa\.0.*}}, metadata ![[var]], metadata !DIExpression(DW_OP_LLVM_fragment, 0, 32), metadata ![[id_4:[0-9]+]], metadata ptr %var.sroa.0, metadata !DIExpression())
; CHECK-NEXT: call void @llvm.dbg.assign(metadata i32 %[[v]], metadata ![[var]], metadata !DIExpression(DW_OP_LLVM_fragment, 32, 32), metadata ![[id_3]], metadata ptr %var.sroa.1, metadata !DIExpression())

; CHECK-DAG: ![[id_1]] = distinct !DIAssignID()
; CHECK-DAG: ![[id_2]] = distinct !DIAssignID()
; CHECK-DAG: ![[id_3]] = distinct !DIAssignID()
; CHECK-DAG: ![[id_4]] = distinct !DIAssignID()

%struct.Pair = type { i32, i32 }

define dso_local noundef i32 @_Z3funv() !dbg !9 {
entry:
  %var = alloca %struct.Pair, align 4, !DIAssignID !19
  call void @llvm.dbg.assign(metadata i1 undef, metadata !14, metadata !DIExpression(), metadata !19, metadata ptr %var, metadata !DIExpression()), !dbg !20
  %ref.tmp = alloca %struct.Pair, align 4
  %call = call i64 @_Z6getVarv(), !dbg !22
  store i64 %call, ptr %ref.tmp, align 4, !dbg !22
  call void @llvm.memcpy.p0.p0.i64(ptr align 4 %var, ptr align 4 %ref.tmp, i64 8, i1 false), !dbg !23, !DIAssignID !29
  call void @llvm.dbg.assign(metadata i1 undef, metadata !14, metadata !DIExpression(DW_OP_LLVM_fragment, 0, 32), metadata !29, metadata ptr %var, metadata !DIExpression()), !dbg !20
  call void @llvm.dbg.assign(metadata i1 undef, metadata !14, metadata !DIExpression(DW_OP_LLVM_fragment, 32, 32), metadata !29, metadata ptr %var, metadata !DIExpression(DW_OP_plus, DW_OP_constu, 4)), !dbg !20
  %b = getelementptr inbounds %struct.Pair, ptr %var, i32 0, i32 1, !dbg !31
  %0 = load i32, ptr %b, align 4, !dbg !31
  ret i32 %0, !dbg !35
}

declare !dbg !36 i64 @_Z6getVarv()
declare void @llvm.dbg.declare(metadata, metadata, metadata)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg)
declare void @llvm.dbg.assign(metadata, metadata, metadata, metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !7}
!llvm.ident = !{!8}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 16.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.cpp", directory: "/")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 8, !"PIC Level", i32 2}
!6 = !{i32 7, !"PIE Level", i32 2}
!7 = !{i32 7, !"uwtable", i32 2}
!8 = !{!"clang version 16.0.0"}
!9 = distinct !DISubprogram(name: "fun", linkageName: "_Z3funv", scope: !1, file: !1, line: 3, type: !10, scopeLine: 3, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !13)
!10 = !DISubroutineType(types: !11)
!11 = !{!12}
!12 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!13 = !{!14}
!14 = !DILocalVariable(name: "var", scope: !9, file: !1, line: 4, type: !15)
!15 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Pair", file: !1, line: 1, size: 64, flags: DIFlagTypePassByValue, elements: !16, identifier: "_ZTS4Pair")
!16 = !{!17, !18}
!17 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !15, file: !1, line: 1, baseType: !12, size: 32)
!18 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !15, file: !1, line: 1, baseType: !12, size: 32, offset: 32)
!19 = distinct !DIAssignID()
!20 = !DILocation(line: 0, scope: !9)
!21 = !DILocation(line: 4, column: 3, scope: !9)
!22 = !DILocation(line: 5, column: 9, scope: !9)
!23 = !DILocation(line: 5, column: 7, scope: !9)
!29 = distinct !DIAssignID()
!30 = !DILocation(line: 5, column: 3, scope: !9)
!31 = !DILocation(line: 6, column: 14, scope: !9)
!34 = !DILocation(line: 7, column: 1, scope: !9)
!35 = !DILocation(line: 6, column: 3, scope: !9)
!36 = !DISubprogram(name: "getVar", linkageName: "_Z6getVarv", scope: !1, file: !1, line: 2, type: !37, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized, retainedNodes: !39)
!37 = !DISubroutineType(types: !38)
!38 = !{!15}
!39 = !{}
