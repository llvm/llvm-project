; RUN: llc -stop-after=finalize-isel %s -o - \
; RUN:    -experimental-assignment-tracking \
; RUN: | FileCheck %s

;; Check that a dbg.assign for a fully stack-homed variable causes the variable
;; location to appear in the Machine Function side table. Similar to
;; single-memory-location.ll except this has slightly more complicated input
;; (there's a loop and an assignment).
;;
;; $ cat test.cpp
;; int get();
;; void esc(int*);
;; void doSomething(int);
;; void fun() {
;;   int local;
;;   esc(&local);
;;   while (local) {
;;     local = get();
;;     doSomething(local);
;;     esc(&local);    
;;   }
;; }
;; $ clang++ -O2 -g -emit-llvm -S -c -Xclang -fexperimental-assignment-tracking

; CHECK: ![[VAR:[0-9]+]] = !DILocalVariable(name: "local",
; CHECK: stack:
; CHECK-NEXT: - { id: 0, name: local, type: default, offset: 0, size: 4, alignment: 4, 
; CHECK-NEXT:     stack-id: default, callee-saved-register: '', callee-saved-restored: true, 
; CHECK-NEXT:     debug-info-variable: '![[VAR]]', debug-info-expression: '!DIExpression()', 
; CHECK-NEXT:     debug-info-location: '!{{.+}}' }

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: uwtable mustprogress
define dso_local void @_Z3funv() local_unnamed_addr #0 !dbg !7 {
entry:
  %local = alloca i32, align 4, !DIAssignID !13
  call void @llvm.dbg.assign(metadata i1 undef, metadata !11, metadata !DIExpression(), metadata !13, metadata ptr %local, metadata !DIExpression()), !dbg !14
  %0 = bitcast ptr %local to ptr, !dbg !15
  call void @llvm.lifetime.start.p0i8(i64 4, ptr nonnull %0) #4, !dbg !15
  call void @_Z3escPi(ptr nonnull %local), !dbg !16
  %1 = load i32, ptr %local, align 4, !dbg !17
  %tobool.not1 = icmp eq i32 %1, 0, !dbg !17
  br i1 %tobool.not1, label %while.end, label %while.body, !dbg !22

while.body:                                       ; preds = %entry, %while.body
  %call = call i32 @_Z3getv(), !dbg !23
  store i32 %call, ptr %local, align 4, !dbg !25, !DIAssignID !26
  call void @llvm.dbg.assign(metadata i32 %call, metadata !11, metadata !DIExpression(), metadata !26, metadata ptr %local, metadata !DIExpression()), !dbg !14
  call void @_Z11doSomethingi(i32 %call), !dbg !27
  call void @_Z3escPi(ptr nonnull %local), !dbg !28
  %2 = load i32, ptr %local, align 4, !dbg !17
  %tobool.not = icmp eq i32 %2, 0, !dbg !17
  br i1 %tobool.not, label %while.end, label %while.body, !dbg !22, !llvm.loop !29

while.end:                                        ; preds = %while.body, %entry
  call void @llvm.lifetime.end.p0i8(i64 4, ptr nonnull %0) #4, !dbg !32
  ret void, !dbg !32
}

declare void @llvm.lifetime.start.p0i8(i64 immarg, ptr nocapture)
declare !dbg !33 dso_local void @_Z3escPi(ptr) local_unnamed_addr
declare !dbg !37 dso_local i32 @_Z3getv() local_unnamed_addr
declare !dbg !40 dso_local void @_Z11doSomethingi(i32) local_unnamed_addr
declare void @llvm.lifetime.end.p0i8(i64 immarg, ptr nocapture)
declare void @llvm.dbg.assign(metadata, metadata, metadata, metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 12.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.cpp", directory: "/")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 12.0.0"}
!7 = distinct !DISubprogram(name: "fun", linkageName: "_Z3funv", scope: !1, file: !1, line: 4, type: !8, scopeLine: 4, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !10)
!8 = !DISubroutineType(types: !9)
!9 = !{null}
!10 = !{!11}
!11 = !DILocalVariable(name: "local", scope: !7, file: !1, line: 5, type: !12)
!12 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!13 = distinct !DIAssignID()
!14 = !DILocation(line: 0, scope: !7)
!15 = !DILocation(line: 5, column: 3, scope: !7)
!16 = !DILocation(line: 6, column: 3, scope: !7)
!17 = !DILocation(line: 7, column: 10, scope: !7)
!22 = !DILocation(line: 7, column: 3, scope: !7)
!23 = !DILocation(line: 8, column: 13, scope: !24)
!24 = distinct !DILexicalBlock(scope: !7, file: !1, line: 7, column: 17)
!25 = !DILocation(line: 8, column: 11, scope: !24)
!26 = distinct !DIAssignID()
!27 = !DILocation(line: 9, column: 5, scope: !24)
!28 = !DILocation(line: 10, column: 5, scope: !24)
!29 = distinct !{!29, !22, !30, !31}
!30 = !DILocation(line: 11, column: 3, scope: !7)
!31 = !{!"llvm.loop.mustprogress"}
!32 = !DILocation(line: 12, column: 1, scope: !7)
!33 = !DISubprogram(name: "esc", linkageName: "_Z3escPi", scope: !1, file: !1, line: 2, type: !34, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized, retainedNodes: !2)
!34 = !DISubroutineType(types: !35)
!35 = !{null, !36}
!36 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !12, size: 64)
!37 = !DISubprogram(name: "get", linkageName: "_Z3getv", scope: !1, file: !1, line: 1, type: !38, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized, retainedNodes: !2)
!38 = !DISubroutineType(types: !39)
!39 = !{!12}
!40 = !DISubprogram(name: "doSomething", linkageName: "_Z11doSomethingi", scope: !1, file: !1, line: 3, type: !41, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized, retainedNodes: !2)
!41 = !DISubroutineType(types: !42)
!42 = !{null, !12}
