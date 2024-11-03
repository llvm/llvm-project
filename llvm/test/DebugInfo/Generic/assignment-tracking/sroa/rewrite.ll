; RUN: opt -passes=sroa,verify -S %s -o - \
; RUN: | FileCheck %s --implicit-check-not="call void @llvm.dbg"

; Check that the new slices of an alloca and memset intructions get dbg.assign
; intrinsics with the correct fragment info. Ensure that only the
; value-expression gets fragment info; that the address-expression remains
; untouched.

;; $ cat test.cpp
;; void do_something();
;; struct LargeStruct {
;;   int A, B, C;
;;   int Var;
;;   int D, E, F;
;; };
;; int Glob;
;; bool Cond;
;; int example() {
;;   LargeStruct S = {0};
;;   S.Var = Glob;
;;   return S.Var;
;; }
;; $ clang test.cpp -Xclang -disable-llvm-passes -O2 -g -c -S -emit-llvm -o -

; CHECK: entry:
; CHECK-NEXT:   %S.sroa.0 = alloca { i32, i32, i32 }, align 8, !DIAssignID ![[ID_1:[0-9]+]]
; CHECK-NEXT:   call void @llvm.dbg.assign(metadata i1 undef, metadata ![[VAR:[0-9]+]], metadata !DIExpression(DW_OP_LLVM_fragment, 0, 96), metadata ![[ID_1]], metadata ptr %S.sroa.0, metadata !DIExpression()), !dbg

;; The middle slice has been promoted, so the alloca has gone away.

; CHECK-NEXT:   %S.sroa.5 = alloca { i32, i32, i32 }, align 8, !DIAssignID ![[ID_3:[0-9]+]]
; CHECK-NEXT:   call void @llvm.dbg.assign(metadata i1 undef, metadata ![[VAR]], metadata !DIExpression(DW_OP_LLVM_fragment, 128, 96), metadata ![[ID_3]], metadata ptr %S.sroa.5, metadata !DIExpression()), !dbg

;; The memset has been sliced up (middle slice removed).
; CHECK: call void @llvm.memset{{.*}}(ptr align 8 %S.sroa.0, i8 0, i64 12, i1 false), !dbg !{{.+}}, !DIAssignID ![[ID_5:[0-9]+]]
; CHECK: call void @llvm.memset{{.*}}(ptr align 8 %S.sroa.5, i8 0, i64 12, i1 false), !dbg !{{.+}}, !DIAssignID ![[ID_6:[0-9]+]]

; CHECK-NEXT: call void @llvm.dbg.assign(metadata i8 0, metadata ![[VAR]], metadata !DIExpression(DW_OP_LLVM_fragment, 0, 96), metadata ![[ID_5]], metadata ptr %S.sroa.0, metadata !DIExpression()), !dbg
;; Check the middle slice (no memset) gets a correct dbg.assign.
; CHECK-NEXT: call void @llvm.dbg.assign(metadata i32 0, metadata ![[VAR]], metadata !DIExpression(DW_OP_LLVM_fragment, 96, 32), metadata !{{.+}}, metadata ptr undef, metadata !DIExpression()), !dbg
; CHECK-NEXT:   call void @llvm.dbg.assign(metadata i8 0, metadata ![[VAR]], metadata !DIExpression(DW_OP_LLVM_fragment, 128, 96), metadata ![[ID_6]], metadata ptr %S.sroa.5, metadata !DIExpression()), !dbg

;; mem2reg promotes the load/store to the middle slice created by SROA:
; CHECK-NEXT: %0 = load i32, ptr @Glob, align 4, !dbg !{{.+}}
; CHECK-NEXT: call void @llvm.dbg.assign(metadata i32 %0, metadata ![[VAR]], metadata !DIExpression(DW_OP_LLVM_fragment, 96, 32), metadata ![[ID_4:[0-9]+]], metadata ptr undef, metadata !DIExpression()), !dbg

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

%struct.LargeStruct = type { i32, i32, i32, i32, i32, i32, i32 }

@Glob = dso_local global i32 0, align 4, !dbg !0
@Cond = dso_local global i8 0, align 1, !dbg !6

; Function Attrs: nounwind uwtable mustprogress
define dso_local i32 @_Z7examplev() #0 !dbg !14 {
entry:
  %S = alloca %struct.LargeStruct, align 4, !DIAssignID !28
  call void @llvm.dbg.assign(metadata i1 undef, metadata !18, metadata !DIExpression(), metadata !28, metadata ptr %S, metadata !DIExpression()), !dbg !29
  %0 = bitcast ptr %S to ptr, !dbg !30
  call void @llvm.lifetime.start.p0i8(i64 28, ptr %0) #4, !dbg !30
  %1 = bitcast ptr %S to ptr, !dbg !31
  call void @llvm.memset.p0i8.i64(ptr align 4 %1, i8 0, i64 28, i1 false), !dbg !31, !DIAssignID !32
  call void @llvm.dbg.assign(metadata i8 0, metadata !18, metadata !DIExpression(), metadata !32, metadata ptr %1, metadata !DIExpression()), !dbg !31
  %2 = load i32, ptr @Glob, align 4, !dbg !33
  %Var = getelementptr inbounds %struct.LargeStruct, ptr %S, i32 0, i32 3, !dbg !38
  store i32 %2, ptr %Var, align 4, !dbg !39, !DIAssignID !42
  call void @llvm.dbg.assign(metadata i32 %2, metadata !18, metadata !DIExpression(DW_OP_LLVM_fragment, 96, 32), metadata !42, metadata ptr %Var, metadata !DIExpression()), !dbg !39
  %Var1 = getelementptr inbounds %struct.LargeStruct, ptr %S, i32 0, i32 3, !dbg !43
  %3 = load i32, ptr %Var1, align 4, !dbg !43
  %4 = bitcast ptr %S to ptr, !dbg !44
  call void @llvm.lifetime.end.p0i8(i64 28, ptr %4) #4, !dbg !44
  ret i32 %3, !dbg !45
}

; Function Attrs: argmemonly nofree nosync nounwind willreturn
declare void @llvm.lifetime.start.p0i8(i64 immarg, ptr nocapture) #1

; Function Attrs: argmemonly nofree nosync nounwind willreturn writeonly
declare void @llvm.memset.p0i8.i64(ptr nocapture writeonly, i8, i64, i1 immarg) #2

; Function Attrs: argmemonly nofree nosync nounwind willreturn
declare void @llvm.lifetime.end.p0i8(i64 immarg, ptr nocapture) #1

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.assign(metadata, metadata, metadata, metadata, metadata, metadata) #3

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!10, !11, !12, !1000}
!llvm.ident = !{!13}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "Glob", scope: !2, file: !3, line: 7, type: !9, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !3, producer: "clang version 12.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "test.cpp", directory: "/")
!4 = !{}
!5 = !{!0, !6}
!6 = !DIGlobalVariableExpression(var: !7, expr: !DIExpression())
!7 = distinct !DIGlobalVariable(name: "Cond", scope: !2, file: !3, line: 8, type: !8, isLocal: false, isDefinition: true)
!8 = !DIBasicType(name: "bool", size: 8, encoding: DW_ATE_boolean)
!9 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!10 = !{i32 7, !"Dwarf Version", i32 4}
!11 = !{i32 2, !"Debug Info Version", i32 3}
!12 = !{i32 1, !"wchar_size", i32 4}
!13 = !{!"clang version 12.0.0"}
!14 = distinct !DISubprogram(name: "example", linkageName: "_Z7examplev", scope: !3, file: !3, line: 9, type: !15, scopeLine: 9, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !17)
!15 = !DISubroutineType(types: !16)
!16 = !{!9}
!17 = !{!18}
!18 = !DILocalVariable(name: "S", scope: !14, file: !3, line: 10, type: !19)
!19 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "LargeStruct", file: !3, line: 2, size: 224, flags: DIFlagTypePassByValue, elements: !20, identifier: "_ZTS11LargeStruct")
!20 = !{!21, !22, !23, !24, !25, !26, !27}
!21 = !DIDerivedType(tag: DW_TAG_member, name: "A", scope: !19, file: !3, line: 3, baseType: !9, size: 32)
!22 = !DIDerivedType(tag: DW_TAG_member, name: "B", scope: !19, file: !3, line: 3, baseType: !9, size: 32, offset: 32)
!23 = !DIDerivedType(tag: DW_TAG_member, name: "C", scope: !19, file: !3, line: 3, baseType: !9, size: 32, offset: 64)
!24 = !DIDerivedType(tag: DW_TAG_member, name: "Var", scope: !19, file: !3, line: 4, baseType: !9, size: 32, offset: 96)
!25 = !DIDerivedType(tag: DW_TAG_member, name: "D", scope: !19, file: !3, line: 5, baseType: !9, size: 32, offset: 128)
!26 = !DIDerivedType(tag: DW_TAG_member, name: "E", scope: !19, file: !3, line: 5, baseType: !9, size: 32, offset: 160)
!27 = !DIDerivedType(tag: DW_TAG_member, name: "F", scope: !19, file: !3, line: 5, baseType: !9, size: 32, offset: 192)
!28 = distinct !DIAssignID()
!29 = !DILocation(line: 0, scope: !14)
!30 = !DILocation(line: 10, column: 3, scope: !14)
!31 = !DILocation(line: 10, column: 15, scope: !14)
!32 = distinct !DIAssignID()
!33 = !DILocation(line: 11, column: 11, scope: !14)
!38 = !DILocation(line: 11, column: 5, scope: !14)
!39 = !DILocation(line: 11, column: 9, scope: !14)
!42 = distinct !DIAssignID()
!43 = !DILocation(line: 12, column: 12, scope: !14)
!44 = !DILocation(line: 13, column: 1, scope: !14)
!45 = !DILocation(line: 12, column: 3, scope: !14)
!1000 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
