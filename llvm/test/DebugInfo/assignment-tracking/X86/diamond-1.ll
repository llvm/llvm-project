; RUN: llc %s -stop-after=finalize-isel -o - \
; RUN: | FileCheck %s


; RUN: llc --try-experimental-debuginfo-iterators %s -stop-after=finalize-isel -o - \
; RUN: | FileCheck %s

;; cat test.cpp
;; void d();
;; void e();
;; void es(int*);
;; int  f(int a) {
;;   if (a) {
;;     e();
;;     a = 100;
;;   } else {
;;     d();
;;     a = 500;
;;   }
;;   es(&a);
;;   return a;
;; }
;; $ clang++ test.cpp -S -emit-llvm -Xclang -fexperimental-assignment-tracking

;; Check that the memory location is selected after the store in if.end:
;; entry:
;;   a = param-value
;; if.then:
;;   a = 100
;; if.else:
;;   a = 500
;; if.end:
;;   store (phi if.then: 100, if.else: 500)
;;   a = in memory

; CHECK-DAG: ![[VAR:[0-9]+]] = !DILocalVariable(name: "a",

; CHECK: bb.0.entry:
; CHECK: DBG_VALUE $edi, $noreg, ![[VAR]], !DIExpression()

; CHECK: bb.1.if.then:
; CHECK: DBG_VALUE 100, $noreg, ![[VAR]], !DIExpression()

; CHECK: bb.2.if.else:
; CHECK: DBG_VALUE 500, $noreg, ![[VAR]], !DIExpression()

; CHECK: bb.3.if.end:
; CHECK-NEXT: %0:gr32 = PHI %2, %bb.1, %3, %bb.2
; CHECK-NEXT: MOV32mr %stack.0.a.addr, 1, $noreg, 0, $noreg, %0
; CHECK-NEXT: DBG_VALUE %stack.0.a.addr, $noreg, ![[VAR]], !DIExpression(DW_OP_deref)

target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: mustprogress uwtable
define dso_local noundef i32 @_Z1fi(i32 noundef %a) local_unnamed_addr #0 !dbg !7 {
entry:
  %a.addr = alloca i32, align 4, !DIAssignID !13
  call void @llvm.dbg.assign(metadata i1 undef, metadata !12, metadata !DIExpression(), metadata !13, metadata ptr %a.addr, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.assign(metadata i32 %a, metadata !12, metadata !DIExpression(), metadata !15, metadata ptr %a.addr, metadata !DIExpression()), !dbg !14
  %tobool.not = icmp eq i32 %a, 0, !dbg !16
  br i1 %tobool.not, label %if.else, label %if.then, !dbg !18

if.then:                                          ; preds = %entry
  tail call void @_Z1ev(), !dbg !19
  call void @llvm.dbg.assign(metadata i32 100, metadata !12, metadata !DIExpression(), metadata !21, metadata ptr %a.addr, metadata !DIExpression()), !dbg !14
  br label %if.end, !dbg !22

if.else:                                          ; preds = %entry
  tail call void @_Z1dv(), !dbg !23
  call void @llvm.dbg.assign(metadata i32 500, metadata !12, metadata !DIExpression(), metadata !21, metadata ptr %a.addr, metadata !DIExpression()), !dbg !14
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  %storemerge = phi i32 [ 500, %if.else ], [ 100, %if.then ], !dbg !25
  store i32 %storemerge, ptr %a.addr, align 4, !dbg !25, !DIAssignID !21
  call void @_Z2esPi(ptr noundef nonnull %a.addr), !dbg !30
  %0 = load i32, ptr %a.addr, align 4, !dbg !31
  ret i32 %0, !dbg !32
}

declare !dbg !33 dso_local void @_Z1ev() local_unnamed_addr #1
declare !dbg !37 dso_local void @_Z1dv() local_unnamed_addr #1
declare !dbg !38 dso_local void @_Z2esPi(ptr noundef) local_unnamed_addr #1
declare void @llvm.dbg.assign(metadata, metadata, metadata, metadata, metadata, metadata) #2

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
!7 = distinct !DISubprogram(name: "f", linkageName: "_Z1fi", scope: !1, file: !1, line: 4, type: !8, scopeLine: 4, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !11)
!8 = !DISubroutineType(types: !9)
!9 = !{!10, !10}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !{!12}
!12 = !DILocalVariable(name: "a", arg: 1, scope: !7, file: !1, line: 4, type: !10)
!13 = distinct !DIAssignID()
!14 = !DILocation(line: 0, scope: !7)
!15 = distinct !DIAssignID()
!16 = !DILocation(line: 5, column: 7, scope: !17)
!17 = distinct !DILexicalBlock(scope: !7, file: !1, line: 5, column: 7)
!18 = !DILocation(line: 5, column: 7, scope: !7)
!19 = !DILocation(line: 6, column: 5, scope: !20)
!20 = distinct !DILexicalBlock(scope: !17, file: !1, line: 5, column: 10)
!21 = distinct !DIAssignID()
!22 = !DILocation(line: 8, column: 3, scope: !20)
!23 = !DILocation(line: 9, column: 5, scope: !24)
!24 = distinct !DILexicalBlock(scope: !17, file: !1, line: 8, column: 10)
!25 = !DILocation(line: 0, scope: !17)
!30 = !DILocation(line: 12, column: 3, scope: !7)
!31 = !DILocation(line: 13, column: 10, scope: !7)
!32 = !DILocation(line: 13, column: 3, scope: !7)
!33 = !DISubprogram(name: "e", linkageName: "_Z1ev", scope: !1, file: !1, line: 2, type: !34, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized, retainedNodes: !36)
!34 = !DISubroutineType(types: !35)
!35 = !{null}
!36 = !{}
!37 = !DISubprogram(name: "d", linkageName: "_Z1dv", scope: !1, file: !1, line: 1, type: !34, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized, retainedNodes: !36)
!38 = !DISubprogram(name: "es", linkageName: "_Z2esPi", scope: !1, file: !1, line: 3, type: !39, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized, retainedNodes: !36)
!39 = !DISubroutineType(types: !40)
!40 = !{null, !41}
!41 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !10, size: 64)
!1000 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
