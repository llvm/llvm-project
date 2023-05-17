; RUN: opt %s -S -passes=instcombine -o - \
; RUN: | FileCheck %s

;; Check that when instcombine sinks an instruction used by a dbg.assign, the
;; usual debug intrinsic updating doesn't take place (i.e. do not
;; clone-and-sink a dbg.assign). Assignment tracking should be able to handle
;; this gracefully for variables that still have a stack home. For fully
;; promoted variables we may need to revisit this.

;; $ cat test.c
;; struct a {
;;   char *b;
;; } c() __attribute__((noreturn));
;; int d;
;; int e();
;; int f() {
;;   if (e())
;;     return d;
;;   c();
;; }
;; void g();
;; void h() {
;;   struct a i;
;;   i.b = 0;
;;   f();
;;   i.b = 0;
;;   g(&i);
;; }
;; $ clang -O2 -g -Xclang -fexperimental-assignment-tracking

; CHECK: f.exit:
; CHECK-NEXT: store ptr null, ptr %i, align 8,{{.+}}, !DIAssignID ![[ID:[0-9]+]]
; CHECK-NEXT: call void @llvm.dbg.assign({{.+}}, {{.+}}, {{.+}}, metadata ![[ID]], metadata ptr %i, {{.+}}), !dbg

%struct.a = type { ptr }

@d = dso_local local_unnamed_addr global i32 0, align 4, !dbg !0

; Function Attrs: nounwind uwtable
define dso_local i32 @f() local_unnamed_addr #0 !dbg !11 {
entry:
  %call = tail call i32 (...) @e() #5, !dbg !14
  %tobool.not = icmp eq i32 %call, 0, !dbg !14
  br i1 %tobool.not, label %if.end, label %if.then, !dbg !16

if.then:                                          ; preds = %entry
  %0 = load i32, ptr @d, align 4, !dbg !17
  ret i32 %0, !dbg !22

if.end:                                           ; preds = %entry
  %call1 = tail call ptr (...) @c() #6, !dbg !23
  unreachable, !dbg !23
}

declare !dbg !24 dso_local i32 @e(...) local_unnamed_addr #1

; Function Attrs: noreturn
declare !dbg !27 dso_local ptr @c(...) local_unnamed_addr #2

; Function Attrs: nounwind uwtable
define dso_local void @h() local_unnamed_addr #0 !dbg !35 {
entry:
  %i = alloca %struct.a, align 8, !DIAssignID !40
  call void @llvm.dbg.assign(metadata i1 undef, metadata !39, metadata !DIExpression(), metadata !40, metadata ptr %i, metadata !DIExpression()), !dbg !41
  %0 = bitcast ptr %i to ptr, !dbg !42
  call void @llvm.lifetime.start.p0i8(i64 8, ptr nonnull %0) #5, !dbg !42
  %b = getelementptr inbounds %struct.a, ptr %i, i64 0, i32 0, !dbg !43
  call void @llvm.dbg.assign(metadata ptr null, metadata !39, metadata !DIExpression(), metadata !44, metadata ptr %b, metadata !DIExpression()), !dbg !41
  %call.i = tail call i32 (...) @e() #5, !dbg !45
  %tobool.not.i = icmp eq i32 %call.i, 0, !dbg !45
  br i1 %tobool.not.i, label %if.end.i, label %f.exit, !dbg !47

if.end.i:                                         ; preds = %entry
  %call1.i = tail call ptr (...) @c() #6, !dbg !48
  unreachable, !dbg !48

f.exit:                                           ; preds = %entry
  store ptr null, ptr %b, align 8, !dbg !49, !DIAssignID !53
  call void @llvm.dbg.assign(metadata ptr null, metadata !39, metadata !DIExpression(), metadata !53, metadata ptr %b, metadata !DIExpression()), !dbg !41
  call void @g(ptr nonnull %i) #5, !dbg !54
  call void @llvm.lifetime.end.p0i8(i64 8, ptr nonnull %0) #5, !dbg !55
  ret void, !dbg !55
}

declare void @llvm.lifetime.start.p0i8(i64 immarg, ptr nocapture) #3
declare dso_local void @g(...) local_unnamed_addr #1
declare void @llvm.lifetime.end.p0i8(i64 immarg, ptr nocapture) #3
declare void @llvm.dbg.assign(metadata, metadata, metadata, metadata, metadata, metadata) #4

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!7, !8, !9, !1000}
!llvm.ident = !{!10}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "d", scope: !2, file: !3, line: 4, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 12.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "test.c", directory: "/")
!4 = !{}
!5 = !{!0}
!6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!7 = !{i32 7, !"Dwarf Version", i32 4}
!8 = !{i32 2, !"Debug Info Version", i32 3}
!9 = !{i32 1, !"wchar_size", i32 4}
!10 = !{!"clang version 12.0.0"}
!11 = distinct !DISubprogram(name: "f", scope: !3, file: !3, line: 6, type: !12, scopeLine: 6, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !4)
!12 = !DISubroutineType(types: !13)
!13 = !{!6}
!14 = !DILocation(line: 7, column: 7, scope: !15)
!15 = distinct !DILexicalBlock(scope: !11, file: !3, line: 7, column: 7)
!16 = !DILocation(line: 7, column: 7, scope: !11)
!17 = !DILocation(line: 8, column: 12, scope: !15)
!22 = !DILocation(line: 8, column: 5, scope: !15)
!23 = !DILocation(line: 9, column: 3, scope: !11)
!24 = !DISubprogram(name: "e", scope: !3, file: !3, line: 5, type: !25, spFlags: DISPFlagOptimized, retainedNodes: !4)
!25 = !DISubroutineType(types: !26)
!26 = !{!6, null}
!27 = !DISubprogram(name: "c", scope: !3, file: !3, line: 3, type: !28, flags: DIFlagNoReturn, spFlags: DISPFlagOptimized, retainedNodes: !4)
!28 = !DISubroutineType(types: !29)
!29 = !{!30, null}
!30 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "a", file: !3, line: 1, size: 64, elements: !31)
!31 = !{!32}
!32 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !30, file: !3, line: 2, baseType: !33, size: 64)
!33 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !34, size: 64)
!34 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!35 = distinct !DISubprogram(name: "h", scope: !3, file: !3, line: 12, type: !36, scopeLine: 12, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !38)
!36 = !DISubroutineType(types: !37)
!37 = !{null}
!38 = !{!39}
!39 = !DILocalVariable(name: "i", scope: !35, file: !3, line: 13, type: !30)
!40 = distinct !DIAssignID()
!41 = !DILocation(line: 0, scope: !35)
!42 = !DILocation(line: 13, column: 3, scope: !35)
!43 = !DILocation(line: 14, column: 5, scope: !35)
!44 = distinct !DIAssignID()
!45 = !DILocation(line: 7, column: 7, scope: !15, inlinedAt: !46)
!46 = distinct !DILocation(line: 15, column: 3, scope: !35)
!47 = !DILocation(line: 7, column: 7, scope: !11, inlinedAt: !46)
!48 = !DILocation(line: 9, column: 3, scope: !11, inlinedAt: !46)
!49 = !DILocation(line: 16, column: 7, scope: !35)
!53 = distinct !DIAssignID()
!54 = !DILocation(line: 17, column: 3, scope: !35)
!55 = !DILocation(line: 18, column: 1, scope: !35)
!1000 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
