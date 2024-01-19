; RUN: opt -passes=sroa,verify -S %s -o - \
; RUN: | FileCheck %s --implicit-check-not="call void @llvm.dbg"
; RUN: opt --try-experimental-debuginfo-iterators -passes=sroa,verify -S %s -o - \
; RUN: | FileCheck %s --implicit-check-not="call void @llvm.dbg"

;; Check that the new slices of an alloca and memcpy intructions get dbg.assign
;; intrinsics with the correct fragment info.
;;
;; Also check that the new dbg.assign intrinsics are inserted after each split
;; store. See llvm/test/DebugInfo/Generic/dbg-assign-sroa-id.ll for the
;; counterpart check. Ensure that only the value-expression gets fragment info;
;; that the address-expression remains untouched.

;; $ cat test.cpp
;; struct LargeStruct {
;;   int A, B, C;
;;   int Var;
;;   int D, E, F;
;; };
;; LargeStruct From;
;; int example() {
;;   LargeStruct To = From;
;;   return To.Var;
;; }
;; $ clang test.cpp -Xclang -fexperimental-assignment-tracking \
;;     -Xclang -disable-llvm-passes -O2 -g -c -S -emit-llvm -o -

;; Split alloca.
; CHECK: entry:
; CHECK-NEXT: %To.sroa.0 = alloca { i32, i32, i32 }, align 8, !DIAssignID ![[ID_1:[0-9]+]]
; CHECK-NEXT: call void @llvm.dbg.assign(metadata {{.+}} undef, metadata ![[TO:[0-9]+]], metadata !DIExpression(DW_OP_LLVM_fragment, 0, 96), metadata ![[ID_1]], metadata ptr %To.sroa.0, metadata !DIExpression()), !dbg

; CHECK-NEXT: %To.sroa.4 = alloca { i32, i32, i32 }, align 8, !DIAssignID ![[ID_3:[0-9]+]]
; CHECK-NEXT: call void @llvm.dbg.assign(metadata {{.+}} undef, metadata ![[TO]], metadata !DIExpression(DW_OP_LLVM_fragment, 128, 96), metadata ![[ID_3]], metadata ptr %To.sroa.4, metadata !DIExpression()), !dbg

;; Split memcpy.
; CHECK: call void @llvm.memcpy{{.*}}(ptr align 8 %To.sroa.0, ptr align 4 @From, i64 12, i1 false),{{.*}}!DIAssignID ![[ID_4:[0-9]+]]
;; This slice has been split and is promoted.
; CHECK: %To.sroa.3.0.copyload = load i32, ptr getelementptr inbounds (i8, ptr @From, i64 12)
; CHECK: call void @llvm.memcpy{{.*}}(ptr align 8 %To.sroa.4, ptr align 4 getelementptr inbounds (i8, ptr @From, i64 16), i64 12, i1 false){{.*}}!DIAssignID ![[ID_6:[0-9]+]]

;; Intrinsics for the splits above.
; CHECK-NEXT: call void @llvm.dbg.assign(metadata {{.+}} undef, metadata ![[TO]], metadata !DIExpression(DW_OP_LLVM_fragment, 0, 96), metadata ![[ID_4]], metadata ptr %To.sroa.0, metadata !DIExpression()), !dbg
; CHECK-NEXT: call void @llvm.dbg.value(metadata i32 %To.sroa.3.0.copyload, metadata ![[TO]], metadata !DIExpression(DW_OP_LLVM_fragment, 96, 32))
; CHECK-NEXT: call void @llvm.dbg.assign(metadata {{.+}} undef, metadata ![[TO]], metadata !DIExpression(DW_OP_LLVM_fragment, 128, 96), metadata ![[ID_6]], metadata ptr %To.sroa.4, metadata !DIExpression()), !dbg

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

%struct.LargeStruct = type { i32, i32, i32, i32, i32, i32, i32 }

@From = dso_local global %struct.LargeStruct zeroinitializer, align 4, !dbg !0

; Function Attrs: nounwind uwtable mustprogress
define dso_local i32 @_Z7examplev() #0 !dbg !20 {
entry:
  %To = alloca %struct.LargeStruct, align 4, !DIAssignID !25
  call void @llvm.dbg.assign(metadata i1 undef, metadata !24, metadata !DIExpression(), metadata !25, metadata ptr %To, metadata !DIExpression()), !dbg !26
  %0 = bitcast ptr %To to ptr, !dbg !27
  call void @llvm.lifetime.start.p0i8(i64 28, ptr %0) #3, !dbg !27
  %1 = bitcast ptr %To to ptr, !dbg !28
  call void @llvm.memcpy.p0i8.p0i8.i64(ptr align 4 %1, ptr align 4 bitcast (ptr @From to ptr), i64 28, i1 false), !dbg !28, !DIAssignID !34
  call void @llvm.dbg.assign(metadata i1 undef, metadata !24, metadata !DIExpression(), metadata !34, metadata ptr %1, metadata !DIExpression()), !dbg !28
  %Var = getelementptr inbounds %struct.LargeStruct, ptr %To, i32 0, i32 3, !dbg !35
  %2 = load i32, ptr %Var, align 4, !dbg !35
  %3 = bitcast ptr %To to ptr, !dbg !38
  call void @llvm.lifetime.end.p0i8(i64 28, ptr %3) #3, !dbg !38
  ret i32 %2, !dbg !39
}

; Function Attrs: argmemonly nofree nosync nounwind willreturn
declare void @llvm.lifetime.start.p0i8(i64 immarg, ptr nocapture) #1

; Function Attrs: argmemonly nofree nosync nounwind willreturn
declare void @llvm.memcpy.p0i8.p0i8.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg) #1

; Function Attrs: argmemonly nofree nosync nounwind willreturn
declare void @llvm.lifetime.end.p0i8(i64 immarg, ptr nocapture) #1

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.assign(metadata, metadata, metadata, metadata, metadata, metadata) #2


!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!16, !17, !18, !1000}
!llvm.ident = !{!19}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "From", scope: !2, file: !3, line: 6, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !3, producer: "clang version 12.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "sroa-test.cpp", directory: "/")
!4 = !{}
!5 = !{!0}
!6 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "LargeStruct", file: !3, line: 1, size: 224, flags: DIFlagTypePassByValue, elements: !7, identifier: "_ZTS11LargeStruct")
!7 = !{!8, !10, !11, !12, !13, !14, !15}
!8 = !DIDerivedType(tag: DW_TAG_member, name: "A", scope: !6, file: !3, line: 2, baseType: !9, size: 32)
!9 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!10 = !DIDerivedType(tag: DW_TAG_member, name: "B", scope: !6, file: !3, line: 2, baseType: !9, size: 32, offset: 32)
!11 = !DIDerivedType(tag: DW_TAG_member, name: "C", scope: !6, file: !3, line: 2, baseType: !9, size: 32, offset: 64)
!12 = !DIDerivedType(tag: DW_TAG_member, name: "Var", scope: !6, file: !3, line: 3, baseType: !9, size: 32, offset: 96)
!13 = !DIDerivedType(tag: DW_TAG_member, name: "D", scope: !6, file: !3, line: 4, baseType: !9, size: 32, offset: 128)
!14 = !DIDerivedType(tag: DW_TAG_member, name: "E", scope: !6, file: !3, line: 4, baseType: !9, size: 32, offset: 160)
!15 = !DIDerivedType(tag: DW_TAG_member, name: "F", scope: !6, file: !3, line: 4, baseType: !9, size: 32, offset: 192)
!16 = !{i32 7, !"Dwarf Version", i32 4}
!17 = !{i32 2, !"Debug Info Version", i32 3}
!18 = !{i32 1, !"wchar_size", i32 4}
!19 = !{!"clang version 12.0.0"}
!20 = distinct !DISubprogram(name: "example", linkageName: "_Z7examplev", scope: !3, file: !3, line: 7, type: !21, scopeLine: 7, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !23)
!21 = !DISubroutineType(types: !22)
!22 = !{!9}
!23 = !{!24}
!24 = !DILocalVariable(name: "To", scope: !20, file: !3, line: 8, type: !6)
!25 = distinct !DIAssignID()
!26 = !DILocation(line: 0, scope: !20)
!27 = !DILocation(line: 8, column: 3, scope: !20)
!28 = !DILocation(line: 8, column: 20, scope: !20)
!34 = distinct !DIAssignID()
!35 = !DILocation(line: 9, column: 13, scope: !20)
!38 = !DILocation(line: 10, column: 1, scope: !20)
!39 = !DILocation(line: 9, column: 3, scope: !20)
!1000 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
