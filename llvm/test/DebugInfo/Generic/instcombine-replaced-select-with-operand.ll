; RUN: opt -passes=instcombine -S %s | FileCheck %s

;; Ensure that debug uses of a select are replaced when the select is replaced
;; with one of its operands in InstCombine's replacedSelectWithOperand.

;; Based on IR generated from this source, compiled with -O2 -g:
;;
;; extern void* glob;
;; struct Thing {
;;   void *ptr;
;;   void *get() {return ptr;}
;; };
;; Thing *two(void *in) {
;;   if ((long)in & 1)
;;     return nullptr;
;;   return static_cast<Thing *>(in);
;; }
;; void one() {
;;   if (Thing *a = two(glob))
;;     // Without the fix we lose 'this' (a) in the inlined call below.
;;     glob = a->get();
;; }

;; Look at the instruction:
;;     %retval.0.i = select i1 %tobool.not.i, ptr %2, ptr null
;; Note that %2 is a bitcast of %0.
;; InstCombine replaces the use of %retval.0.i in if.then with %0 because
;; if.then is only branched to when retval.0.is not null. Ensure that debug use
;; of %retval.0.i is also replaced with %0.

; CHECK: if.then:
; CHECK-NEXT: #dbg_value(ptr %0, ![[THIS:[0-9]+]], !DIExpression(),
; CHECK: ![[THIS]] = !DILocalVariable(name: "this",

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.Thing = type { ptr }

@glob = external dso_local local_unnamed_addr global ptr, align 8

define dso_local void @_Z3onev() local_unnamed_addr !dbg !29 {
entry:
  %0 = load ptr, ptr @glob, align 8, !dbg !35
  call void @llvm.dbg.value(metadata ptr %0, metadata !22, metadata !DIExpression()), !dbg !40
  %1 = ptrtoint ptr %0 to i64, !dbg !42
  %and.i = and i64 %1, 1, !dbg !43
  %tobool.not.i = icmp eq i64 %and.i, 0, !dbg !42
  %2 = bitcast ptr %0 to ptr, !dbg !44
  %retval.0.i = select i1 %tobool.not.i, ptr %2, ptr null, !dbg !44
  call void @llvm.dbg.value(metadata ptr %retval.0.i, metadata !33, metadata !DIExpression()), !dbg !45
  %tobool.not = icmp eq ptr %retval.0.i, null, !dbg !46
  br i1 %tobool.not, label %if.end, label %if.then, !dbg !47

if.then:                                          ; preds = %entry
  call void @llvm.dbg.value(metadata ptr %retval.0.i, metadata !48, metadata !DIExpression()), !dbg !51
  %ptr.i = getelementptr inbounds %struct.Thing, ptr %retval.0.i, i64 0, i32 0, !dbg !53
  %3 = load ptr, ptr %ptr.i, align 8, !dbg !53
  store ptr %3, ptr @glob, align 8, !dbg !56
  br label %if.end, !dbg !57

if.end:                                           ; preds = %if.then, %entry
  ret void, !dbg !58
}

declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!14, !15, !16}
!llvm.ident = !{!17}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 13.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !3, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.cpp", directory: "/")
!2 = !{}
!3 = !{!4, !5}
!4 = !DIBasicType(name: "long int", size: 64, encoding: DW_ATE_signed)
!5 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !6, size: 64)
!6 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Thing", file: !1, line: 2, size: 64, flags: DIFlagTypePassByValue, elements: !7, identifier: "_ZTS5Thing")
!7 = !{!8, !10}
!8 = !DIDerivedType(tag: DW_TAG_member, name: "ptr", scope: !6, file: !1, line: 3, baseType: !9, size: 64)
!9 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!10 = !DISubprogram(name: "get", linkageName: "_ZN5Thing3getEv", scope: !6, file: !1, line: 4, type: !11, scopeLine: 4, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!11 = !DISubroutineType(types: !12)
!12 = !{!9, !13}
!13 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !6, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!14 = !{i32 7, !"Dwarf Version", i32 4}
!15 = !{i32 2, !"Debug Info Version", i32 3}
!16 = !{i32 1, !"wchar_size", i32 4}
!17 = !{!"clang version 13.0.0"}
!18 = distinct !DISubprogram(name: "two", linkageName: "_Z3twoPv", scope: !1, file: !1, line: 6, type: !19, scopeLine: 6, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !21)
!19 = !DISubroutineType(types: !20)
!20 = !{!5, !9}
!21 = !{!22}
!22 = !DILocalVariable(name: "in", arg: 1, scope: !18, file: !1, line: 6, type: !9)
!23 = !DILocation(line: 0, scope: !18)
!24 = !DILocation(line: 7, column: 7, scope: !25)
!25 = distinct !DILexicalBlock(scope: !18, file: !1, line: 7, column: 7)
!26 = !DILocation(line: 7, column: 16, scope: !25)
!27 = !DILocation(line: 7, column: 7, scope: !18)
!28 = !DILocation(line: 10, column: 1, scope: !18)
!29 = distinct !DISubprogram(name: "one", linkageName: "_Z3onev", scope: !1, file: !1, line: 11, type: !30, scopeLine: 11, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !32)
!30 = !DISubroutineType(types: !31)
!31 = !{null}
!32 = !{!33}
!33 = !DILocalVariable(name: "a", scope: !34, file: !1, line: 12, type: !5)
!34 = distinct !DILexicalBlock(scope: !29, file: !1, line: 12, column: 14)
!35 = !DILocation(line: 12, column: 22, scope: !34)
!40 = !DILocation(line: 0, scope: !18, inlinedAt: !41)
!41 = distinct !DILocation(line: 12, column: 18, scope: !34)
!42 = !DILocation(line: 7, column: 7, scope: !25, inlinedAt: !41)
!43 = !DILocation(line: 7, column: 16, scope: !25, inlinedAt: !41)
!44 = !DILocation(line: 7, column: 7, scope: !18, inlinedAt: !41)
!45 = !DILocation(line: 0, scope: !34)
!46 = !DILocation(line: 12, column: 14, scope: !34)
!47 = !DILocation(line: 12, column: 14, scope: !29)
!48 = !DILocalVariable(name: "this", arg: 1, scope: !49, type: !5, flags: DIFlagArtificial | DIFlagObjectPointer)
!49 = distinct !DISubprogram(name: "get", linkageName: "_ZN5Thing3getEv", scope: !6, file: !1, line: 4, type: !11, scopeLine: 4, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, declaration: !10, retainedNodes: !50)
!50 = !{!48}
!51 = !DILocation(line: 0, scope: !49, inlinedAt: !52)
!52 = distinct !DILocation(line: 14, column: 15, scope: !34)
!53 = !DILocation(line: 4, column: 23, scope: !49, inlinedAt: !52)
!56 = !DILocation(line: 14, column: 10, scope: !34)
!57 = !DILocation(line: 14, column: 5, scope: !34)
!58 = !DILocation(line: 15, column: 1, scope: !29)
!59 = !DILocation(line: 0, scope: !49)
!60 = !DILocation(line: 4, column: 23, scope: !49)
!61 = !DILocation(line: 4, column: 16, scope: !49)
