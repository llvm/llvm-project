; RUN: opt -passes=inline %s -S -o - \
; RUN: | FileCheck %s
; RUN: opt --try-experimental-debuginfo-iterators -passes=inline %s -S -o - \
; RUN: | FileCheck %s

;; $ cat test.cpp
;; __attribute__((always_inline))
;; static void a(int* p2, int v2) { *p2 = v2; }
;;
;; __attribute__((always_inline))
;; void b(int* p1, int v1) { a(p1, v1); }
;;
;; int f1() {
;;   int f1_local;
;;   a(&f1_local, 1);
;;   return f1_local;
;; }
;;
;; int f2() {
;;   int f2_local[2];
;;   a(f2_local, 2);
;;   return f2_local[0];
;; }
;;
;; int f3() {
;;   int f3_local[2];
;;   a(f3_local + 1, 3);
;;   return f3_local[1];
;; }
;;
;; int f4(int f4_param) {
;;   a(&f4_param, 4);
;;   return f4_param;
;; }
;;
;; int f5(int f5_param) {
;;   int &f5_alias = f5_param;
;;   a(&f5_alias, 5);
;;   return f5_param;
;; }
;;
;; int f6() {
;;   int f6_local;
;;   b(&f6_local, 6);
;;   return f6_local;
;; }
;;
;; IR generated with:
;; $ clang++ -Xclang -disable-llvm-passes test.cpp -g -O2 -o - -S -emit-llvm \
;;   | opt -passes=declare-to-assign,sroa -o - -S
;;
;; Check that inlined stores are tracked as assignments. FileCheck directives
;; inline.

source_filename = "test.cpp"
define dso_local void @_Z1bPii(ptr %p1, i32 %v1) #0 !dbg !7 {
entry:
  call void @llvm.dbg.assign(metadata i1 undef, metadata !13, metadata !DIExpression(), metadata !15, metadata ptr undef, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.assign(metadata i1 undef, metadata !14, metadata !DIExpression(), metadata !17, metadata ptr undef, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.assign(metadata ptr %p1, metadata !13, metadata !DIExpression(), metadata !18, metadata ptr undef, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.assign(metadata i32 %v1, metadata !14, metadata !DIExpression(), metadata !19, metadata ptr undef, metadata !DIExpression()), !dbg !16
  call void @_ZL1aPii(ptr %p1, i32 %v1), !dbg !20
  ret void, !dbg !21
}

define internal void @_ZL1aPii(ptr %p2, i32 %v2) #2 !dbg !22 {
entry:
  call void @llvm.dbg.assign(metadata i1 undef, metadata !24, metadata !DIExpression(), metadata !26, metadata ptr undef, metadata !DIExpression()), !dbg !27
  call void @llvm.dbg.assign(metadata i1 undef, metadata !25, metadata !DIExpression(), metadata !28, metadata ptr undef, metadata !DIExpression()), !dbg !27
  call void @llvm.dbg.assign(metadata ptr %p2, metadata !24, metadata !DIExpression(), metadata !29, metadata ptr undef, metadata !DIExpression()), !dbg !27
  call void @llvm.dbg.assign(metadata i32 %v2, metadata !25, metadata !DIExpression(), metadata !30, metadata ptr undef, metadata !DIExpression()), !dbg !27
  store i32 %v2, ptr %p2, align 4, !dbg !31
  ret void, !dbg !36
}

;; Store directly to local with one level of inlining. Also record f1_dbg to
;; check that the dbg.assign gets the correct scope after inlining. This check
;; isn't repeated for the other functions.
;; int f1() {
;;   int f1_local;
;;   a(&f1_local, 1);
;;   return f1_local;
;; }
;;
; CHECK-LABEL: define dso_local i32 @_Z2f1v()
; CHECK:       store i32 1, ptr %f1_local, align 4,{{.*}} !DIAssignID ![[ID_1:[0-9]+]]
; CHECK-NEXT:  #dbg_assign(i32 1, ![[f1_local:[0-9]+]], !DIExpression(), ![[ID_1]], ptr %f1_local, !DIExpression(), ![[f1_dbg:[0-9]+]]
define dso_local i32 @_Z2f1v() #3 !dbg !37 {
entry:
  %f1_local = alloca i32, align 4, !DIAssignID !42
  call void @llvm.dbg.assign(metadata i1 undef, metadata !41, metadata !DIExpression(), metadata !42, metadata ptr %f1_local, metadata !DIExpression()), !dbg !43
  %0 = bitcast ptr %f1_local to ptr, !dbg !44
  call void @llvm.lifetime.start.p0(i64 4, ptr %0) #5, !dbg !44
  call void @_ZL1aPii(ptr %f1_local, i32 1), !dbg !45
  %1 = load i32, ptr %f1_local, align 4, !dbg !46
  %2 = bitcast ptr %f1_local to ptr, !dbg !47
  call void @llvm.lifetime.end.p0(i64 4, ptr %2) #5, !dbg !47
  ret i32 %1, !dbg !48
}

;; Store directly to fragment of local at its base address.
;; int f2() {
;;   int f2_local[2];
;;   a(f2_local, 2);
;;   return f2_local[0];
;; }
;;
; CHECK-LABEL: define dso_local i32 @_Z2f2v()
; CHECK:       store i32 2, ptr %arraydecay, align 4,{{.*}} !DIAssignID ![[ID_2:[0-9]+]]
; CHECK-NEXT:  #dbg_assign(i32 2, ![[f2_local:[0-9]+]], !DIExpression(DW_OP_LLVM_fragment, 0, 32), ![[ID_2]], ptr %arraydecay, !DIExpression(),
define dso_local i32 @_Z2f2v() #3 !dbg !49 {
entry:
  %f2_local = alloca [2 x i32], align 4, !DIAssignID !55
  call void @llvm.dbg.assign(metadata i1 undef, metadata !51, metadata !DIExpression(), metadata !55, metadata ptr %f2_local, metadata !DIExpression()), !dbg !56
  %0 = bitcast ptr %f2_local to ptr, !dbg !57
  call void @llvm.lifetime.start.p0(i64 8, ptr %0) #5, !dbg !57
  %arraydecay = getelementptr inbounds [2 x i32], ptr %f2_local, i64 0, i64 0, !dbg !58
  call void @_ZL1aPii(ptr %arraydecay, i32 2), !dbg !59
  %arrayidx = getelementptr inbounds [2 x i32], ptr %f2_local, i64 0, i64 0, !dbg !60
  %1 = load i32, ptr %arrayidx, align 4, !dbg !60
  %2 = bitcast ptr %f2_local to ptr, !dbg !61
  call void @llvm.lifetime.end.p0(i64 8, ptr %2) #5, !dbg !61
  ret i32 %1, !dbg !62
}

;; Store to fragment of local using rvalue offset as argument.
;; int f3() {
;;   int f3_local[2];
;;   a(f3_local + 1, 3);
;;   return f3_local[1];
;; }
; CHECK-LABEL: define dso_local i32 @_Z2f3v()
; CHECK:       store i32 3, ptr %add.ptr, align 4,{{.*}} !DIAssignID ![[ID_3:[0-9]+]]
; CHECK-NEXT:  #dbg_assign(i32 3, ![[f3_local:[0-9]+]], !DIExpression(DW_OP_LLVM_fragment, 32, 32), ![[ID_3]], ptr %add.ptr, !DIExpression(),
define dso_local i32 @_Z2f3v() #3 !dbg !63 {
entry:
  %f3_local = alloca [2 x i32], align 4, !DIAssignID !66
  call void @llvm.dbg.assign(metadata i1 undef, metadata !65, metadata !DIExpression(), metadata !66, metadata ptr %f3_local, metadata !DIExpression()), !dbg !67
  %0 = bitcast ptr %f3_local to ptr, !dbg !68
  call void @llvm.lifetime.start.p0(i64 8, ptr %0) #5, !dbg !68
  %arraydecay = getelementptr inbounds [2 x i32], ptr %f3_local, i64 0, i64 0, !dbg !69
  %add.ptr = getelementptr inbounds i32, ptr %arraydecay, i64 1, !dbg !70
  call void @_ZL1aPii(ptr %add.ptr, i32 3), !dbg !71
  %arrayidx = getelementptr inbounds [2 x i32], ptr %f3_local, i64 0, i64 1, !dbg !72
  %1 = load i32, ptr %arrayidx, align 4, !dbg !72
  %2 = bitcast ptr %f3_local to ptr, !dbg !73
  call void @llvm.lifetime.end.p0(i64 8, ptr %2) #5, !dbg !73
  ret i32 %1, !dbg !74
}

;; Store to parameter directly.
;; int f4(int f4_param) {
;;   a(&f4_param, 4);
;;   return f4_param;
;; }
; CHECK-LABEL: define dso_local i32 @_Z2f4i(i32 %f4_param)
; CHECK:       store i32 4, ptr %f4_param.addr, align 4,{{.*}} !DIAssignID ![[ID_4:[0-9]+]]
; CHECK-NEXT:  #dbg_assign(i32 4, ![[f4_param:[0-9]+]], !DIExpression(), ![[ID_4]], ptr %f4_param.addr, !DIExpression(),
define dso_local i32 @_Z2f4i(i32 %f4_param) #3 !dbg !75 {
entry:
  %f4_param.addr = alloca i32, align 4, !DIAssignID !80
  call void @llvm.dbg.assign(metadata i1 undef, metadata !79, metadata !DIExpression(), metadata !80, metadata ptr %f4_param.addr, metadata !DIExpression()), !dbg !81
  store i32 %f4_param, ptr %f4_param.addr, align 4, !DIAssignID !82
  call void @llvm.dbg.assign(metadata i32 %f4_param, metadata !79, metadata !DIExpression(), metadata !82, metadata ptr %f4_param.addr, metadata !DIExpression()), !dbg !81
  call void @_ZL1aPii(ptr %f4_param.addr, i32 4), !dbg !83
  %0 = load i32, ptr %f4_param.addr, align 4, !dbg !84
  ret i32 %0, !dbg !85
}

;; Store through an alias.
;; int f5(int f5_param) {
;;   int &f5_alias = f5_param;
;;   a(&f5_alias, 5);
;;   return f5_param;
;; }
; CHECK-LABEL: define dso_local i32 @_Z2f5i(i32 %f5_param)
; CHECK:       store i32 5, ptr %f5_param.addr, align 4,{{.*}}!DIAssignID ![[ID_5:[0-9]+]]
; CHECK-NEXT:  #dbg_assign(i32 5, ![[f5_param:[0-9]+]], !DIExpression(), ![[ID_5]], ptr %f5_param.addr, !DIExpression(),
define dso_local i32 @_Z2f5i(i32 %f5_param) #3 !dbg !86 {
entry:
  %f5_param.addr = alloca i32, align 4, !DIAssignID !91
  call void @llvm.dbg.assign(metadata i1 undef, metadata !88, metadata !DIExpression(), metadata !91, metadata ptr %f5_param.addr, metadata !DIExpression()), !dbg !92
  call void @llvm.dbg.assign(metadata i1 undef, metadata !89, metadata !DIExpression(), metadata !93, metadata ptr undef, metadata !DIExpression()), !dbg !92
  store i32 %f5_param, ptr %f5_param.addr, align 4, !DIAssignID !94
  call void @llvm.dbg.assign(metadata i32 %f5_param, metadata !88, metadata !DIExpression(), metadata !94, metadata ptr %f5_param.addr, metadata !DIExpression()), !dbg !92
  call void @llvm.dbg.assign(metadata ptr %f5_param.addr, metadata !89, metadata !DIExpression(), metadata !95, metadata ptr undef, metadata !DIExpression()), !dbg !92
  call void @_ZL1aPii(ptr %f5_param.addr, i32 5), !dbg !96
  %0 = load i32, ptr %f5_param.addr, align 4, !dbg !97
  ret i32 %0, !dbg !98
}

;; int f6() {
;;   int f6_local;
;;   b(&f6_local, 6);
;;   return f6_local;
;; }
; CHECK-LABEL: define dso_local i32 @_Z2f6v()
; CHECK:       store i32 6, ptr %f6_local, align 4,{{.*}} !DIAssignID ![[ID_6:[0-9]+]]
; CHECK-NEXT:  #dbg_assign(i32 6, ![[f6_local:[0-9]+]], !DIExpression(), ![[ID_6]], ptr %f6_local, !DIExpression(),
define dso_local i32 @_Z2f6v() #3 !dbg !99 {
entry:
  %f6_local = alloca i32, align 4, !DIAssignID !102
  call void @llvm.dbg.assign(metadata i1 undef, metadata !101, metadata !DIExpression(), metadata !102, metadata ptr %f6_local, metadata !DIExpression()), !dbg !103
  %0 = bitcast ptr %f6_local to ptr, !dbg !104
  call void @llvm.lifetime.start.p0(i64 4, ptr %0) #5, !dbg !104
  call void @_Z1bPii(ptr %f6_local, i32 6), !dbg !105
  %1 = load i32, ptr %f6_local, align 4, !dbg !106
  %2 = bitcast ptr %f6_local to ptr, !dbg !107
  call void @llvm.lifetime.end.p0(i64 4, ptr %2) #5, !dbg !107
  ret i32 %1, !dbg !108
}

declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture)
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture)
declare void @llvm.dbg.assign(metadata, metadata, metadata, metadata, metadata, metadata)

; CHECK-DAG: ![[f1_local]] = !DILocalVariable(name: "f1_local",
; CHECK-DAG: ![[f2_local]] = !DILocalVariable(name: "f2_local",
; CHECK-DAG: ![[f3_local]] = !DILocalVariable(name: "f3_local",
; CHECK-DAG: ![[f4_param]] = !DILocalVariable(name: "f4_param",
; CHECK-DAG: ![[f5_param]] = !DILocalVariable(name: "f5_param",
; CHECK-DAG: ![[f6_local]] = !DILocalVariable(name: "f6_local",

; CHECK-DAG: ![[f1_dbg]] = !DILocation(line: 0, scope: ![[f1_scope:[0-9]+]])
; CHECK-DAG: ![[f1_scope]] = distinct !DISubprogram(name: "f1",

; CHECK-DAG: [[ID_1]] = distinct !DIAssignID()
; CHECK-DAG: [[ID_2]] = distinct !DIAssignID()
; CHECK-DAG: [[ID_3]] = distinct !DIAssignID()
; CHECK-DAG: [[ID_4]] = distinct !DIAssignID()
; CHECK-DAG: [[ID_5]] = distinct !DIAssignID()
; CHECK-DAG: [[ID_6]] = distinct !DIAssignID()

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !1000}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 12.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.cpp", directory: "/")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 12.0.0)"}
!7 = distinct !DISubprogram(name: "b", linkageName: "_Z1bPii", scope: !1, file: !1, line: 5, type: !8, scopeLine: 5, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !12)
!8 = !DISubroutineType(types: !9)
!9 = !{null, !10, !11}
!10 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !11, size: 64)
!11 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!12 = !{!13, !14}
!13 = !DILocalVariable(name: "p1", arg: 1, scope: !7, file: !1, line: 5, type: !10)
!14 = !DILocalVariable(name: "v1", arg: 2, scope: !7, file: !1, line: 5, type: !11)
!15 = distinct !DIAssignID()
!16 = !DILocation(line: 0, scope: !7)
!17 = distinct !DIAssignID()
!18 = distinct !DIAssignID()
!19 = distinct !DIAssignID()
!20 = !DILocation(line: 5, column: 27, scope: !7)
!21 = !DILocation(line: 5, column: 38, scope: !7)
!22 = distinct !DISubprogram(name: "a", linkageName: "_ZL1aPii", scope: !1, file: !1, line: 2, type: !8, scopeLine: 2, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !23)
!23 = !{!24, !25}
!24 = !DILocalVariable(name: "p2", arg: 1, scope: !22, file: !1, line: 2, type: !10)
!25 = !DILocalVariable(name: "v2", arg: 2, scope: !22, file: !1, line: 2, type: !11)
!26 = distinct !DIAssignID()
!27 = !DILocation(line: 0, scope: !22)
!28 = distinct !DIAssignID()
!29 = distinct !DIAssignID()
!30 = distinct !DIAssignID()
!31 = !DILocation(line: 2, column: 38, scope: !22)
!36 = !DILocation(line: 2, column: 44, scope: !22)
!37 = distinct !DISubprogram(name: "f1", linkageName: "_Z2f1v", scope: !1, file: !1, line: 7, type: !38, scopeLine: 7, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !40)
!38 = !DISubroutineType(types: !39)
!39 = !{!11}
!40 = !{!41}
!41 = !DILocalVariable(name: "f1_local", scope: !37, file: !1, line: 8, type: !11)
!42 = distinct !DIAssignID()
!43 = !DILocation(line: 0, scope: !37)
!44 = !DILocation(line: 8, column: 3, scope: !37)
!45 = !DILocation(line: 9, column: 3, scope: !37)
!46 = !DILocation(line: 10, column: 10, scope: !37)
!47 = !DILocation(line: 11, column: 1, scope: !37)
!48 = !DILocation(line: 10, column: 3, scope: !37)
!49 = distinct !DISubprogram(name: "f2", linkageName: "_Z2f2v", scope: !1, file: !1, line: 13, type: !38, scopeLine: 13, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !50)
!50 = !{!51}
!51 = !DILocalVariable(name: "f2_local", scope: !49, file: !1, line: 14, type: !52)
!52 = !DICompositeType(tag: DW_TAG_array_type, baseType: !11, size: 64, elements: !53)
!53 = !{!54}
!54 = !DISubrange(count: 2)
!55 = distinct !DIAssignID()
!56 = !DILocation(line: 0, scope: !49)
!57 = !DILocation(line: 14, column: 3, scope: !49)
!58 = !DILocation(line: 15, column: 5, scope: !49)
!59 = !DILocation(line: 15, column: 3, scope: !49)
!60 = !DILocation(line: 16, column: 10, scope: !49)
!61 = !DILocation(line: 17, column: 1, scope: !49)
!62 = !DILocation(line: 16, column: 3, scope: !49)
!63 = distinct !DISubprogram(name: "f3", linkageName: "_Z2f3v", scope: !1, file: !1, line: 19, type: !38, scopeLine: 19, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !64)
!64 = !{!65}
!65 = !DILocalVariable(name: "f3_local", scope: !63, file: !1, line: 20, type: !52)
!66 = distinct !DIAssignID()
!67 = !DILocation(line: 0, scope: !63)
!68 = !DILocation(line: 20, column: 3, scope: !63)
!69 = !DILocation(line: 21, column: 5, scope: !63)
!70 = !DILocation(line: 21, column: 14, scope: !63)
!71 = !DILocation(line: 21, column: 3, scope: !63)
!72 = !DILocation(line: 22, column: 10, scope: !63)
!73 = !DILocation(line: 23, column: 1, scope: !63)
!74 = !DILocation(line: 22, column: 3, scope: !63)
!75 = distinct !DISubprogram(name: "f4", linkageName: "_Z2f4i", scope: !1, file: !1, line: 25, type: !76, scopeLine: 25, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !78)
!76 = !DISubroutineType(types: !77)
!77 = !{!11, !11}
!78 = !{!79}
!79 = !DILocalVariable(name: "f4_param", arg: 1, scope: !75, file: !1, line: 25, type: !11)
!80 = distinct !DIAssignID()
!81 = !DILocation(line: 0, scope: !75)
!82 = distinct !DIAssignID()
!83 = !DILocation(line: 26, column: 3, scope: !75)
!84 = !DILocation(line: 27, column: 10, scope: !75)
!85 = !DILocation(line: 27, column: 3, scope: !75)
!86 = distinct !DISubprogram(name: "f5", linkageName: "_Z2f5i", scope: !1, file: !1, line: 30, type: !76, scopeLine: 30, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !87)
!87 = !{!88, !89}
!88 = !DILocalVariable(name: "f5_param", arg: 1, scope: !86, file: !1, line: 30, type: !11)
!89 = !DILocalVariable(name: "f5_alias", scope: !86, file: !1, line: 31, type: !90)
!90 = !DIDerivedType(tag: DW_TAG_reference_type, baseType: !11, size: 64)
!91 = distinct !DIAssignID()
!92 = !DILocation(line: 0, scope: !86)
!93 = distinct !DIAssignID()
!94 = distinct !DIAssignID()
!95 = distinct !DIAssignID()
!96 = !DILocation(line: 32, column: 3, scope: !86)
!97 = !DILocation(line: 33, column: 10, scope: !86)
!98 = !DILocation(line: 33, column: 3, scope: !86)
!99 = distinct !DISubprogram(name: "f6", linkageName: "_Z2f6v", scope: !1, file: !1, line: 36, type: !38, scopeLine: 36, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !100)
!100 = !{!101}
!101 = !DILocalVariable(name: "f6_local", scope: !99, file: !1, line: 37, type: !11)
!102 = distinct !DIAssignID()
!103 = !DILocation(line: 0, scope: !99)
!104 = !DILocation(line: 37, column: 3, scope: !99)
!105 = !DILocation(line: 38, column: 3, scope: !99)
!106 = !DILocation(line: 39, column: 10, scope: !99)
!107 = !DILocation(line: 40, column: 1, scope: !99)
!108 = !DILocation(line: 39, column: 3, scope: !99)
!1000 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
