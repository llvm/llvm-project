; RUN: opt -inline -S < %s | FileCheck %s
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
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-apple-darwin"

%struct.A = type { i32, [2 x double] }

@a = global %struct.A zeroinitializer, align 8
@b = global %struct.A zeroinitializer, align 8

; Function Attrs: nounwind
declare void @_Z3fn31A(%struct.A* nocapture readonly %p1) #0

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture readonly, i64, i32, i1) #2

; Function Attrs: nounwind
define void @_Z3fn4v() #0 !dbg !21 {
entry:
; Test that the dbg.declare is moved together with the alloca.
; CHECK: define void @_Z3fn5v()
; CHECK-NEXT: entry:
; CHECK-NEXT:   %agg.tmp.sroa.3.i = alloca [20 x i8], align 4
; CHECK-NEXT:   call void @llvm.dbg.declare(metadata [20 x i8]* %agg.tmp.sroa.3.i,
  %agg.tmp.sroa.3 = alloca [20 x i8], align 4
  tail call void @llvm.dbg.declare(metadata [20 x i8]* %agg.tmp.sroa.3, metadata !46, metadata !48), !dbg !49
  %agg.tmp.sroa.0.0.copyload = load i32, i32* getelementptr inbounds (%struct.A, %struct.A* @b, i64 0, i32 0), align 8, !dbg !50
  tail call void @llvm.dbg.value(metadata i32 %agg.tmp.sroa.0.0.copyload, i64 0, metadata !46, metadata !51), !dbg !49
  %agg.tmp.sroa.3.0..sroa_idx = getelementptr inbounds [20 x i8], [20 x i8]* %agg.tmp.sroa.3, i64 0, i64 0, !dbg !50
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %agg.tmp.sroa.3.0..sroa_idx, i8* getelementptr (i8, i8* bitcast (%struct.A* @b to i8*), i64 4), i64 20, i32 4, i1 false), !dbg !50
  tail call void @llvm.dbg.declare(metadata %struct.A* undef, metadata !46, metadata !31) #2, !dbg !49
  %tobool.i = icmp eq i32 %agg.tmp.sroa.0.0.copyload, 0, !dbg !52
  br i1 %tobool.i, label %_Z3fn31A.exit, label %if.then.i, !dbg !53

if.then.i:                                        ; preds = %entry
  store i32 %agg.tmp.sroa.0.0.copyload, i32* getelementptr inbounds (%struct.A, %struct.A* @a, i64 0, i32 0), align 8, !dbg !54
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* getelementptr (i8, i8* bitcast (%struct.A* @a to i8*), i64 4), i8* %agg.tmp.sroa.3.0..sroa_idx, i64 20, i32 4, i1 false), !dbg !54
  br label %_Z3fn31A.exit, !dbg !54

_Z3fn31A.exit:                                    ; preds = %entry, %if.then.i
  ret void, !dbg !50
}

; Function Attrs: noreturn nounwind
define void @_Z3fn5v() #3 !dbg !24 {
entry:
  br label %while.body, !dbg !55

while.body:                                       ; preds = %entry, %while.body
  call void @_Z3fn4v(), !dbg !56
  br label %while.body, !dbg !55
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind }
attributes #3 = { noreturn nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!28, !29}
!llvm.ident = !{!30}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.7.0 (trunk 227480) (llvm/trunk 227517)", isOptimized: true, emissionKind: FullDebug, file: !1, enums: !2, retainedTypes: !3, globals: !25, imports: !2)
!1 = !DIFile(filename: "<stdin>", directory: "")
!2 = !{}
!3 = !{!4}
!4 = !DICompositeType(tag: DW_TAG_structure_type, name: "A", line: 1, size: 192, align: 64, file: !5, elements: !6, identifier: "_ZTS1A")
!5 = !DIFile(filename: "test.cpp", directory: "")
!6 = !{!7, !9}
!7 = !DIDerivedType(tag: DW_TAG_member, name: "arg0", line: 2, size: 32, align: 32, file: !5, scope: !4, baseType: !8)
!8 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!9 = !DIDerivedType(tag: DW_TAG_member, name: "arg1", line: 3, size: 128, align: 64, offset: 64, file: !5, scope: !4, baseType: !10)
!10 = !DICompositeType(tag: DW_TAG_array_type, size: 128, align: 64, baseType: !11, elements: !12)
!11 = !DIBasicType(tag: DW_TAG_base_type, name: "double", size: 64, align: 64, encoding: DW_ATE_float)
!12 = !{!13}
!13 = !DISubrange(count: 2)
!15 = distinct !DISubprogram(name: "fn3", linkageName: "_Z3fn31A", line: 6, isLocal: false, isDefinition: true, flags: DIFlagPrototyped, isOptimized: true, unit: !0, scopeLine: 6, file: !5, scope: !16, type: !17, variables: !19)
!16 = !DIFile(filename: "test.cpp", directory: "")
!17 = !DISubroutineType(types: !18)
!18 = !{null, !4}
!19 = !{!20}
!20 = !DILocalVariable(name: "p1", line: 6, arg: 1, scope: !15, file: !16, type: !4)
!21 = distinct !DISubprogram(name: "fn4", linkageName: "_Z3fn4v", line: 11, isLocal: false, isDefinition: true, flags: DIFlagPrototyped, isOptimized: true, unit: !0, scopeLine: 11, file: !5, scope: !16, type: !22, variables: !2)
!22 = !DISubroutineType(types: !23)
!23 = !{null}
!24 = distinct !DISubprogram(name: "fn5", linkageName: "_Z3fn5v", line: 13, isLocal: false, isDefinition: true, flags: DIFlagPrototyped, isOptimized: true, unit: !0, scopeLine: 13, file: !5, scope: !16, type: !22, variables: !2)
!25 = !{!26, !27}
!26 = !DIGlobalVariable(name: "a", line: 4, isLocal: false, isDefinition: true, scope: null, file: !16, type: !4, variable: %struct.A* @a)
!27 = !DIGlobalVariable(name: "b", line: 4, isLocal: false, isDefinition: true, scope: null, file: !16, type: !4, variable: %struct.A* @b)
!28 = !{i32 2, !"Dwarf Version", i32 4}
!29 = !{i32 2, !"Debug Info Version", i32 3}
!30 = !{!"clang version 3.7.0 (trunk 227480) (llvm/trunk 227517)"}
!31 = !DIExpression(DW_OP_deref)
!32 = !DILocation(line: 6, scope: !15)
!33 = !DILocation(line: 7, scope: !34)
!34 = distinct !DILexicalBlock(line: 7, column: 0, file: !5, scope: !15)
!35 = !{!36, !37, i64 0}
!36 = !{!4, !37, i64 0, !38, i64 8}
!37 = !{!"int", !38, i64 0}
!38 = !{!"omnipotent char", !39, i64 0}
!39 = !{!"Simple C/C++ TBAA"}
!40 = !DILocation(line: 7, scope: !15)
!41 = !DILocation(line: 8, scope: !34)
!42 = !{i64 0, i64 4, !43, i64 8, i64 16, !44}
!43 = !{!37, !37, i64 0}
!44 = !{!38, !38, i64 0}
!45 = !DILocation(line: 9, scope: !15)
!46 = !DILocalVariable(name: "p1", line: 6, arg: 1, scope: !15, file: !16, type: !4)
!47 = distinct !DILocation(line: 11, scope: !21)
!48 = !DIExpression(DW_OP_bit_piece, 32, 160)
!49 = !DILocation(line: 6, scope: !15, inlinedAt: !47)
!50 = !DILocation(line: 11, scope: !21)
!51 = !DIExpression(DW_OP_bit_piece, 0, 32)
!52 = !DILocation(line: 7, scope: !34, inlinedAt: !47)
!53 = !DILocation(line: 7, scope: !15, inlinedAt: !47)
!54 = !DILocation(line: 8, scope: !34, inlinedAt: !47)
!55 = !DILocation(line: 14, scope: !24)
!56 = !DILocation(line: 15, scope: !24)
