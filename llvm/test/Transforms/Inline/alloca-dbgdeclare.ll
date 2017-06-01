; RUN: opt -inline -S < %s | FileCheck %s
; RUN: opt -passes='cgscc(inline)' -S < %s | FileCheck %s
; struct A {
;   int arg0;
;   double arg1[2];
; } a, b;
;  
; void fn3(A p1) {
;   if (p1.arg0)
;     a = p1;
; }
;  
; void fn4() { fn3(b); }
;  
; void fn5() {
;   while (1)
;     fn4();
; }
; ModuleID = 'test.cpp'
source_filename = "test/Transforms/Inline/alloca-dbgdeclare.ll"
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-apple-darwin"

%struct.A = type { i32, [2 x double] }

@a = global %struct.A zeroinitializer, align 8, !dbg !0
@b = global %struct.A zeroinitializer, align 8, !dbg !12

; Function Attrs: nounwind
declare void @_Z3fn31A(%struct.A* nocapture readonly) #0

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i32, i1) #2

; Function Attrs: nounwind
define void @_Z3fn4v() #0 !dbg !22 {
entry:
; Test that the dbg.declare is moved together with the alloca.
; CHECK: define void @_Z3fn5v()
; CHECK-NEXT: entry:
; CHECK-NEXT:   %agg.tmp.sroa.3.i = alloca [20 x i8], align 4
; CHECK-NEXT:   call void @llvm.dbg.declare(metadata [20 x i8]* %agg.tmp.sroa.3.i,
  %agg.tmp.sroa.3 = alloca [20 x i8], align 4
  tail call void @llvm.dbg.declare(metadata [20 x i8]* %agg.tmp.sroa.3, metadata !25, metadata !30), !dbg !31
  %agg.tmp.sroa.0.0.copyload = load i32, i32* getelementptr inbounds (%struct.A, %struct.A* @b, i64 0, i32 0), align 8, !dbg !33
  tail call void @llvm.dbg.value(metadata i32 %agg.tmp.sroa.0.0.copyload, i64 0, metadata !25, metadata !34), !dbg !31
  %agg.tmp.sroa.3.0..sroa_idx = getelementptr inbounds [20 x i8], [20 x i8]* %agg.tmp.sroa.3, i64 0, i64 0, !dbg !33
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %agg.tmp.sroa.3.0..sroa_idx, i8* getelementptr (i8, i8* bitcast (%struct.A* @b to i8*), i64 4), i64 20, i32 4, i1 false), !dbg !33
  tail call void @llvm.dbg.declare(metadata %struct.A* undef, metadata !25, metadata !35) #0, !dbg !31
  %tobool.i = icmp eq i32 %agg.tmp.sroa.0.0.copyload, 0, !dbg !36
  br i1 %tobool.i, label %_Z3fn31A.exit, label %if.then.i, !dbg !38

if.then.i:                                        ; preds = %entry
  store i32 %agg.tmp.sroa.0.0.copyload, i32* getelementptr inbounds (%struct.A, %struct.A* @a, i64 0, i32 0), align 8, !dbg !39
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* getelementptr (i8, i8* bitcast (%struct.A* @a to i8*), i64 4), i8* %agg.tmp.sroa.3.0..sroa_idx, i64 20, i32 4, i1 false), !dbg !39
  br label %_Z3fn31A.exit, !dbg !39

_Z3fn31A.exit:                                    ; preds = %if.then.i, %entry

  ret void, !dbg !33
}

; Function Attrs: noreturn nounwind
define void @_Z3fn5v() #3 !dbg !40 {
entry:
  br label %while.body, !dbg !41

while.body:                                       ; preds = %while.body, %entry
  call void @_Z3fn4v(), !dbg !42
  br label %while.body, !dbg !41
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
attributes #2 = { argmemonly nounwind }
attributes #3 = { noreturn nounwind }

!llvm.dbg.cu = !{!14}
!llvm.module.flags = !{!19, !20}
!llvm.ident = !{!21}

!0 = !DIGlobalVariableExpression(var: !1)
!1 = !DIGlobalVariable(name: "a", scope: null, file: !2, line: 4, type: !3, isLocal: false, isDefinition: true)
!2 = !DIFile(filename: "test.cpp", directory: "")
!3 = !DICompositeType(tag: DW_TAG_structure_type, name: "A", file: !2, line: 1, size: 192, align: 64, elements: !4, identifier: "_ZTS1A")
!4 = !{!5, !7}
!5 = !DIDerivedType(tag: DW_TAG_member, name: "arg0", scope: !3, file: !2, line: 2, baseType: !6, size: 32, align: 32)
!6 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!7 = !DIDerivedType(tag: DW_TAG_member, name: "arg1", scope: !3, file: !2, line: 3, baseType: !8, size: 128, align: 64, offset: 64)
!8 = !DICompositeType(tag: DW_TAG_array_type, baseType: !9, size: 128, align: 64, elements: !10)
!9 = !DIBasicType(name: "double", size: 64, align: 64, encoding: DW_ATE_float)
!10 = !{!11}
!11 = !DISubrange(count: 2)
!12 = !DIGlobalVariableExpression(var: !13)
!13 = !DIGlobalVariable(name: "b", scope: null, file: !2, line: 4, type: !3, isLocal: false, isDefinition: true)
!14 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !15, producer: "clang version 3.7.0 (trunk 227480) (llvm/trunk 227517)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !16, retainedTypes: !17, globals: !18, imports: !16)
!15 = !DIFile(filename: "<stdin>", directory: "")
!16 = !{}
!17 = !{!3}
!18 = !{!0, !12}
!19 = !{i32 2, !"Dwarf Version", i32 4}
!20 = !{i32 2, !"Debug Info Version", i32 3}
!21 = !{!"clang version 3.7.0 (trunk 227480) (llvm/trunk 227517)"}
!22 = distinct !DISubprogram(name: "fn4", linkageName: "_Z3fn4v", scope: !2, file: !2, line: 11, type: !23, isLocal: false, isDefinition: true, scopeLine: 11, flags: DIFlagPrototyped, isOptimized: true, unit: !14, variables: !16)
!23 = !DISubroutineType(types: !24)
!24 = !{null}
!25 = !DILocalVariable(name: "p1", arg: 1, scope: !26, file: !2, line: 6, type: !3)
!26 = distinct !DISubprogram(name: "fn3", linkageName: "_Z3fn31A", scope: !2, file: !2, line: 6, type: !27, isLocal: false, isDefinition: true, scopeLine: 6, flags: DIFlagPrototyped, isOptimized: true, unit: !14, variables: !29)
!27 = !DISubroutineType(types: !28)
!28 = !{null, !3}
!29 = !{!25}
!30 = !DIExpression(DW_OP_LLVM_fragment, 32, 160)
!31 = !DILocation(line: 6, scope: !26, inlinedAt: !32)
!32 = distinct !DILocation(line: 11, scope: !22)
!33 = !DILocation(line: 11, scope: !22)
!34 = !DIExpression(DW_OP_LLVM_fragment, 0, 32)
!35 = !DIExpression(DW_OP_deref)
!36 = !DILocation(line: 7, scope: !37, inlinedAt: !32)
!37 = distinct !DILexicalBlock(scope: !26, file: !2, line: 7)
!38 = !DILocation(line: 7, scope: !26, inlinedAt: !32)
!39 = !DILocation(line: 8, scope: !37, inlinedAt: !32)
!40 = distinct !DISubprogram(name: "fn5", linkageName: "_Z3fn5v", scope: !2, file: !2, line: 13, type: !23, isLocal: false, isDefinition: true, scopeLine: 13, flags: DIFlagPrototyped, isOptimized: true, unit: !14, variables: !16)
!41 = !DILocation(line: 14, scope: !40)
!42 = !DILocation(line: 15, scope: !40)

