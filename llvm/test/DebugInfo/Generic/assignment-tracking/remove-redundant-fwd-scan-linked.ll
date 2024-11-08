; RUN: opt -passes=redundant-dbg-inst-elim -S %s -o - \
; RUN: | FileCheck %s --implicit-check-not="call void @llvm.dbg"
; RUN: opt --try-experimental-debuginfo-iterators -passes=redundant-dbg-inst-elim -S %s -o - \
; RUN: | FileCheck %s --implicit-check-not="call void @llvm.dbg"

;; $ cat -n reduce.c
;;  1	void ext();
;;  2	typedef struct {
;;  3	  short *a;
;;  4	  char *b;
;;  5	} c;
;;  6	char d;
;;  7	void f();
;;  8	typedef struct {
;;  9	  c decoder;
;; 10	} e;
;; 11	void g() {
;; 12	  e a;
;; 13	  (&(&a)->decoder)->b = 0;
;; 14	  (&(&a)->decoder)->a = 0;
;; 15	  ext();
;; 16	  a.decoder.b = &d;
;; 17	  f(&a);
;; 18	}

;; clang -O2 -g -Xclang -fexperiemental-assignment-tracking:
;;
;; MemCpyOptPass: Aggregate scalar stores from line 13 and 14 into memset.
;; DSE: Shorten aggregated store because field 'b' is re-assigned later.
;;      Insert an unlinked dbg.assign after the dbg.assign describing
;;      the fragment for field 'b', which is still linked to the memset.
;;
;; Check we don't delete that inserted unlinked dbg.assign.

; CHECK:      %a = alloca %struct.e, align 8, !DIAssignID ![[ID_0:[0-9]+]]
; CHECK-NEXT: #dbg_assign({{.*}}, ![[ID_0]],{{.*}})

;; This dbg.assign is linked to the memset.
; CHECK:      #dbg_assign(ptr null,{{.*}}, !DIExpression(DW_OP_LLVM_fragment, 64, 64), ![[ID_1:[0-9]+]], ptr %b, !DIExpression(),

;; Importantly, check this unlinked dbg.assign which is shadowed by the
;; dbg.assign above isn't deleted.
; CHECK-NEXT: #dbg_assign(ptr null,{{.*}}, !DIExpression(DW_OP_LLVM_fragment, 64, 64), ![[ID_2:[0-9]+]], ptr undef, !DIExpression(),

; CHECK:      #dbg_assign(ptr null,{{.*}}, !DIExpression(DW_OP_LLVM_fragment, 0, 64), ![[ID_1]], ptr %a2, !DIExpression(),

; CHECK:      call void @llvm.memset{{.*}}, !DIAssignID ![[ID_1]]

; CHECK:      store ptr @d, ptr %b, align 8,{{.*}}!DIAssignID ![[ID_3:[0-9]+]]
; CHECK-NEXT: #dbg_assign(ptr @d,{{.*}}, !DIExpression(DW_OP_LLVM_fragment, 64, 64), ![[ID_3]], ptr %b, !DIExpression(),

; CHECK-DAG: ![[ID_0]] = distinct !DIAssignID()
; CHECK-DAG: ![[ID_1]] = distinct !DIAssignID()
; CHECK-DAG: ![[ID_2]] = distinct !DIAssignID()
; CHECK-DAG: ![[ID_3]] = distinct !DIAssignID()

%struct.e = type { %struct.c }
%struct.c = type { ptr, ptr }

@d = dso_local global i8 0, align 1, !dbg !0

define dso_local void @g() local_unnamed_addr #0 !dbg !12 {
entry:
  %a = alloca %struct.e, align 8, !DIAssignID !29
  call void @llvm.dbg.assign(metadata i1 undef, metadata !16, metadata !DIExpression(), metadata !29, metadata ptr %a, metadata !DIExpression()), !dbg !30
  call void @llvm.lifetime.start.p0(i64 16, ptr nonnull %a) #5, !dbg !31
  %b = getelementptr inbounds %struct.e, ptr %a, i64 0, i32 0, i32 1, !dbg !32
  call void @llvm.dbg.assign(metadata ptr null, metadata !16, metadata !DIExpression(DW_OP_LLVM_fragment, 64, 64), metadata !33, metadata ptr %b, metadata !DIExpression()), !dbg !30
  call void @llvm.dbg.assign(metadata ptr null, metadata !16, metadata !DIExpression(DW_OP_LLVM_fragment, 64, 64), metadata !34, metadata ptr undef, metadata !DIExpression()), !dbg !30
  %a2 = getelementptr inbounds %struct.e, ptr %a, i64 0, i32 0, i32 0, !dbg !35
  call void @llvm.dbg.assign(metadata ptr null, metadata !16, metadata !DIExpression(DW_OP_LLVM_fragment, 0, 64), metadata !33, metadata ptr %a2, metadata !DIExpression()), !dbg !30
  call void @llvm.memset.p0.i64(ptr align 8 %a2, i8 0, i64 8, i1 false), !dbg !35, !DIAssignID !33
  tail call void (...) @ext() #5, !dbg !36
  store ptr @d, ptr %b, align 8, !dbg !37, !DIAssignID !44
  call void @llvm.dbg.assign(metadata ptr @d, metadata !16, metadata !DIExpression(DW_OP_LLVM_fragment, 64, 64), metadata !44, metadata ptr %b, metadata !DIExpression()), !dbg !30
  call void @llvm.lifetime.end.p0(i64 16, ptr nonnull %a) #5, !dbg !46
  ret void, !dbg !46
}

declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture) #1
declare !dbg !47 dso_local void @ext(...) local_unnamed_addr #2
declare dso_local void @f(...) local_unnamed_addr #2
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture) #1
declare void @llvm.dbg.assign(metadata, metadata, metadata, metadata, metadata, metadata) #3
declare void @llvm.memset.p0.i64(ptr nocapture writeonly, i8, i64, i1 immarg) #4

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!6, !7, !8, !9, !10, !1000}
!llvm.ident = !{!11}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "d", scope: !2, file: !3, line: 6, type: !5, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 16.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "reduce.c", directory: "/")
!4 = !{!0}
!5 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!6 = !{i32 7, !"Dwarf Version", i32 5}
!7 = !{i32 2, !"Debug Info Version", i32 3}
!8 = !{i32 1, !"wchar_size", i32 4}
!9 = !{i32 7, !"uwtable", i32 1}
!10 = !{i32 7, !"frame-pointer", i32 2}
!11 = !{!"clang version 14.0.0"}
!12 = distinct !DISubprogram(name: "g", scope: !3, file: !3, line: 11, type: !13, scopeLine: 11, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !15)
!13 = !DISubroutineType(types: !14)
!14 = !{null}
!15 = !{!16}
!16 = !DILocalVariable(name: "a", scope: !12, file: !3, line: 12, type: !17)
!17 = !DIDerivedType(tag: DW_TAG_typedef, name: "e", file: !3, line: 10, baseType: !18)
!18 = distinct !DICompositeType(tag: DW_TAG_structure_type, file: !3, line: 8, size: 128, elements: !19)
!19 = !{!20}
!20 = !DIDerivedType(tag: DW_TAG_member, name: "decoder", scope: !18, file: !3, line: 9, baseType: !21, size: 128)
!21 = !DIDerivedType(tag: DW_TAG_typedef, name: "c", file: !3, line: 5, baseType: !22)
!22 = distinct !DICompositeType(tag: DW_TAG_structure_type, file: !3, line: 2, size: 128, elements: !23)
!23 = !{!24, !27}
!24 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !22, file: !3, line: 3, baseType: !25, size: 64)
!25 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !26, size: 64)
!26 = !DIBasicType(name: "short", size: 16, encoding: DW_ATE_signed)
!27 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !22, file: !3, line: 4, baseType: !28, size: 64, offset: 64)
!28 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !5, size: 64)
!29 = distinct !DIAssignID()
!30 = !DILocation(line: 0, scope: !12)
!31 = !DILocation(line: 12, scope: !12)
!32 = !DILocation(line: 13, scope: !12)
!33 = distinct !DIAssignID()
!34 = distinct !DIAssignID()
!35 = !DILocation(line: 14, scope: !12)
!36 = !DILocation(line: 15, scope: !12)
!37 = !DILocation(line: 16, scope: !12)
!44 = distinct !DIAssignID()
!45 = !DILocation(line: 17, scope: !12)
!46 = !DILocation(line: 18, scope: !12)
!47 = !DISubprogram(name: "ext", scope: !3, file: !3, line: 1, type: !13, spFlags: DISPFlagOptimized, retainedNodes: !48)
!48 = !{}
!1000 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
