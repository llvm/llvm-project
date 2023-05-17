; RUN: opt %s -S -passes=instcombine | FileCheck %s

;; Check that instcombine merges the DIAssignID metadata when merging two
;; stores into a successor. Filecheck directives inline.
;;
;; Generated from the following source:
;; int c;
;; void esc(int*);
;; int get();
;; void fun() {
;;   int local;
;;   if (c) {
;;     get();
;;     local = 2;
;;   } else {
;;     local = 2;
;;   }
;;   esc(&local);
;; }

; CHECK: if.then:
; CHECK-NEXT: %call = call
; CHECK-NEXT: call void @llvm.dbg.assign(metadata i32 2, metadata ![[LOCAL:[0-9]+]], metadata !DIExpression(), metadata ![[MERGED_ID:[0-9]+]], metadata ptr %local, metadata !DIExpression()), !dbg

; CHECK: if.else:
; CHECK-NEXT: call void @llvm.dbg.assign(metadata i32 2, metadata ![[LOCAL]], metadata !DIExpression(), metadata ![[MERGED_ID]], metadata ptr %local, metadata !DIExpression()), !dbg

; CHECK: if.end:
; CHECK-NEXT: store i32 2, ptr %local{{.*}}!DIAssignID ![[MERGED_ID]]

; CHECK: ![[LOCAL]] = !DILocalVariable(name: "local",

@c = dso_local local_unnamed_addr global i32 0, align 4, !dbg !0

; Function Attrs: uwtable mustprogress
define dso_local void @_Z3funv() local_unnamed_addr !dbg !11 {
entry:
  %local = alloca i32, align 4
  %0 = bitcast ptr %local to ptr, !dbg !16
  call void @llvm.lifetime.start.p0i8(i64 4, ptr %0), !dbg !16
  %1 = load i32, ptr @c, align 4, !dbg !17
  %tobool = icmp ne i32 %1, 0, !dbg !17
  br i1 %tobool, label %if.then, label %if.else, !dbg !23

if.then:                                          ; preds = %entry
  %call = call i32 @_Z3getv(), !dbg !24
  store i32 2, ptr %local, align 4, !dbg !26, !DIAssignID !27
  call void @llvm.dbg.assign(metadata i32 2, metadata !15, metadata !DIExpression(), metadata !27, metadata ptr %local, metadata !DIExpression()), !dbg !26
  br label %if.end, !dbg !28

if.else:                                          ; preds = %entry
  store i32 2, ptr %local, align 4, !dbg !29, !DIAssignID !31
  call void @llvm.dbg.assign(metadata i32 2, metadata !15, metadata !DIExpression(), metadata !31, metadata ptr %local, metadata !DIExpression()), !dbg !29
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  call void @_Z3escPi(ptr %local), !dbg !32
  call void @llvm.lifetime.end.p0i8(i64 4, ptr %0), !dbg !33
  ret void, !dbg !33
}

declare !dbg !34 dso_local i32 @_Z3getv() local_unnamed_addr
declare !dbg !37 dso_local void @_Z3escPi(ptr) local_unnamed_addr
declare void @llvm.lifetime.start.p0i8(i64 immarg, ptr nocapture)
declare void @llvm.lifetime.end.p0i8(i64 immarg, ptr nocapture)
declare void @llvm.dbg.assign(metadata, metadata, metadata, metadata, metadata, metadata)

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!7, !8, !9, !1000}
!llvm.ident = !{!10}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "c", scope: !2, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !3, producer: "clang version 12.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "test.cpp", directory: "/")
!4 = !{}
!5 = !{!0}
!6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!7 = !{i32 7, !"Dwarf Version", i32 4}
!8 = !{i32 2, !"Debug Info Version", i32 3}
!9 = !{i32 1, !"wchar_size", i32 4}
!10 = !{!"clang version 12.0.0"}
!11 = distinct !DISubprogram(name: "fun", linkageName: "_Z3funv", scope: !3, file: !3, line: 4, type: !12, scopeLine: 4, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !14)
!12 = !DISubroutineType(types: !13)
!13 = !{null}
!14 = !{!15}
!15 = !DILocalVariable(name: "local", scope: !11, file: !3, line: 5, type: !6)
!16 = !DILocation(line: 5, column: 3, scope: !11)
!17 = !DILocation(line: 6, column: 7, scope: !18)
!18 = distinct !DILexicalBlock(scope: !11, file: !3, line: 6, column: 7)
!23 = !DILocation(line: 6, column: 7, scope: !11)
!24 = !DILocation(line: 7, column: 5, scope: !25)
!25 = distinct !DILexicalBlock(scope: !18, file: !3, line: 6, column: 10)
!26 = !DILocation(line: 8, column: 11, scope: !25)
!27 = distinct !DIAssignID()
!28 = !DILocation(line: 9, column: 3, scope: !25)
!29 = !DILocation(line: 10, column: 11, scope: !30)
!30 = distinct !DILexicalBlock(scope: !18, file: !3, line: 9, column: 10)
!31 = distinct !DIAssignID()
!32 = !DILocation(line: 12, column: 3, scope: !11)
!33 = !DILocation(line: 13, column: 1, scope: !11)
!34 = !DISubprogram(name: "get", linkageName: "_Z3getv", scope: !3, file: !3, line: 3, type: !35, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized, retainedNodes: !4)
!35 = !DISubroutineType(types: !36)
!36 = !{!6}
!37 = !DISubprogram(name: "esc", linkageName: "_Z3escPi", scope: !3, file: !3, line: 2, type: !38, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized, retainedNodes: !4)
!38 = !DISubroutineType(types: !39)
!39 = !{null, !40}
!40 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !6, size: 64)
!1000 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
