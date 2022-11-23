; RUN: opt -passes=simplifycfg %s -S -experimental-assignment-tracking  \
; RUN: | FileCheck %s

;; Ensure that we correctly update the value component of dbg.assign intrinsics
;; after merging a conditional block with a store its the predecessor. The
;; value stored is still conditional, but the store itself is now
;; unconditionally run, so we must be sure that any linked dbg.assign intrinsics
;; are tracking the new stored value (the result of the select). If we don't,
;; and the store were to be removed by another pass (e.g. DSE), then we'd
;; eventually end up emitting a location describing the conditional value,
;; unconditionally.

;; Created from the following source and command, with dbg.assign and DIAssignID
;; metadata added and some other metadata removed by hand:
;; $ cat test.c
;; int a;
;; void b() {
;;   int c = 0;
;;   if (a)
;;      c = 1;
;; }
;; $ clang -O2 -g -emit-llvm -S test.c -Xclang -fexperimental-assignment-tracking

; CHECK: %[[SELECT:.*]] = select i1 %tobool
; CHECK-NEXT: store i32 %[[SELECT]], ptr %c{{.*}}, !DIAssignID ![[ID:[0-9]+]]
; CHECK-NEXT: call void @llvm.dbg.assign(metadata i32 %[[SELECT]], metadata ![[VAR_C:[0-9]+]], metadata !DIExpression(), metadata ![[ID]], metadata ptr %c, metadata !DIExpression()), !dbg
; CHECK: ![[VAR_C]] = !DILocalVariable(name: "c",

@a = dso_local global i32 0, align 4, !dbg !0

define dso_local void @b() !dbg !11 {
entry:
  %c = alloca i32, align 4
  %0 = bitcast ptr %c to ptr, !dbg !16
  call void @llvm.lifetime.start.p0i8(i64 4, ptr %0), !dbg !16
  store i32 0, ptr %c, align 4, !dbg !17, !DIAssignID !36
  call void @llvm.dbg.assign(metadata i32 0, metadata !15, metadata !DIExpression(), metadata !36, metadata ptr %c, metadata !DIExpression()), !dbg !17
  %1 = load i32, ptr @a, align 4, !dbg !22
  %tobool = icmp ne i32 %1, 0, !dbg !22
  br i1 %tobool, label %if.then, label %if.end, !dbg !24

if.then:                                          ; preds = %entry
  store i32 1, ptr %c, align 4, !dbg !25, !DIAssignID !37
  call void @llvm.dbg.assign(metadata i32 1, metadata !15, metadata !DIExpression(), metadata !37, metadata ptr %c, metadata !DIExpression()), !dbg !17
  br label %if.end, !dbg !26

if.end:                                           ; preds = %if.then, %entry
  %2 = bitcast ptr %c to ptr, !dbg !27
  call void @llvm.lifetime.end.p0i8(i64 4, ptr %2), !dbg !27
  ret void, !dbg !27
}

declare void @llvm.lifetime.start.p0i8(i64 immarg, ptr nocapture)
declare void @llvm.dbg.assign(metadata, metadata, metadata, metadata, metadata, metadata)
declare void @llvm.lifetime.end.p0i8(i64 immarg, ptr nocapture)

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!7, !8, !9}
!llvm.ident = !{!10}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "a", scope: !2, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 14.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "test.c", directory: "/")
!4 = !{}
!5 = !{!0}
!6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!7 = !{i32 7, !"Dwarf Version", i32 4}
!8 = !{i32 2, !"Debug Info Version", i32 3}
!9 = !{i32 1, !"wchar_size", i32 4}
!10 = !{!"clang version 12.0.0"}
!11 = distinct !DISubprogram(name: "b", scope: !3, file: !3, line: 2, type: !12, scopeLine: 2, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !14)
!12 = !DISubroutineType(types: !13)
!13 = !{null}
!14 = !{!15}
!15 = !DILocalVariable(name: "c", scope: !11, file: !3, line: 3, type: !6)
!16 = !DILocation(line: 3, column: 3, scope: !11)
!17 = !DILocation(line: 3, column: 7, scope: !11)
!22 = !DILocation(line: 4, column: 7, scope: !23)
!23 = distinct !DILexicalBlock(scope: !11, file: !3, line: 4, column: 7)
!24 = !DILocation(line: 4, column: 7, scope: !11)
!25 = !DILocation(line: 5, column: 7, scope: !23)
!26 = !DILocation(line: 5, column: 5, scope: !23)
!27 = !DILocation(line: 6, column: 1, scope: !11)
!36 = distinct !DIAssignID()
!37 = distinct !DIAssignID()
