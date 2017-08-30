; RUN: llc -disable-fp-elim -O0 -filetype=obj < %s | llvm-dwarfdump - | FileCheck %s
; Test that a variable with multiple entries in the MMI table makes it into the
; debug info.
;
; CHECK: DW_TAG_inlined_subroutine
; CHECK:    "_Z3f111A"
; CHECK: DW_TAG_formal_parameter
; CHECK: DW_AT_location [DW_FORM_block1]    (DW_OP_piece 0x1, DW_OP_fbreg -47, DW_OP_piece 0xf, DW_OP_piece 0x1, DW_OP_fbreg -54, DW_OP_piece 0x7)
; CHECK: DW_AT_abstract_origin {{.*}} "p1"
;
; long a;
; struct A {
;   bool x4;
;   void *x5;
;   bool x6;
; };
; int *b;
; struct B {
;   B(long);
;   ~B();
; };
; void f9(A);
; void f13(A p1) {
;   b = (int *)__builtin_operator_new(a);
;   f9(p1);
; }
; void f11(A p1) { f13(p1); }
; void f16() {
;   A c;
;   B d(a);
;   c.x6 = c.x4 = true;
;   f11(c);
; }
; ModuleID = 'test.cpp'
source_filename = "test/DebugInfo/AArch64/frameindices.ll"
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-apple-ios"

%struct.A = type { i8, i8*, i8 }
%struct.B = type { i8 }

@a = global i64 0, align 8, !dbg !0
@b = global i32* null, align 8, !dbg !4

define void @_Z3f131A(%struct.A* nocapture readonly %p1) !dbg !32 {
entry:
  %agg.tmp = alloca %struct.A, align 8
  tail call void @llvm.dbg.declare(metadata %struct.A* %p1, metadata !36, metadata !37), !dbg !38
  %0 = load i64, i64* @a, align 8, !dbg !39, !tbaa !40
  %call = tail call noalias i8* @_Znwm(i64 %0) #4, !dbg !44
  store i8* %call, i8** bitcast (i32** @b to i8**), align 8, !dbg !45, !tbaa !46
  %1 = getelementptr inbounds %struct.A, %struct.A* %agg.tmp, i64 0, i32 0, !dbg !48
  %2 = getelementptr inbounds %struct.A, %struct.A* %p1, i64 0, i32 0, !dbg !48
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %1, i8* %2, i64 24, i32 8, i1 false), !dbg !48, !tbaa.struct !49
  call void @_Z2f91A(%struct.A* %agg.tmp), !dbg !52
  ret void, !dbg !53
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #0

; Function Attrs: nobuiltin
declare noalias i8* @_Znwm(i64) #1

declare void @_Z2f91A(%struct.A*)

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i32, i1) #2

define void @_Z3f111A(%struct.A* nocapture readonly %p1) !dbg !54 {
entry:
  %agg.tmp.i = alloca %struct.A, align 8
  tail call void @llvm.dbg.declare(metadata %struct.A* %p1, metadata !56, metadata !37), !dbg !57
  %0 = getelementptr inbounds %struct.A, %struct.A* %p1, i64 0, i32 0, !dbg !58
  %1 = getelementptr inbounds %struct.A, %struct.A* %agg.tmp.i, i64 0, i32 0, !dbg !59
  call void @llvm.lifetime.start(i64 24, i8* %1), !dbg !59
  %2 = load i64, i64* @a, align 8, !dbg !61, !tbaa !40
  %call.i = tail call noalias i8* @_Znwm(i64 %2) #4, !dbg !62
  store i8* %call.i, i8** bitcast (i32** @b to i8**), align 8, !dbg !63, !tbaa !46
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %1, i8* %0, i64 24, i32 8, i1 false), !dbg !64
  call void @_Z2f91A(%struct.A* %agg.tmp.i), !dbg !65
  call void @llvm.lifetime.end(i64 24, i8* %1), !dbg !66
  ret void, !dbg !67
}

define void @_Z3f16v() personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) !dbg !68 {
entry:
  %agg.tmp.i.i = alloca %struct.A, align 8
  %d = alloca %struct.B, align 1
  %agg.tmp.sroa.2 = alloca [15 x i8], align 1
  %agg.tmp.sroa.4 = alloca [7 x i8], align 1
  tail call void @llvm.dbg.declare(metadata [15 x i8]* %agg.tmp.sroa.2, metadata !56, metadata !74), !dbg !75
  tail call void @llvm.dbg.declare(metadata [7 x i8]* %agg.tmp.sroa.4, metadata !56, metadata !77), !dbg !75
  tail call void @llvm.dbg.declare(metadata %struct.A* undef, metadata !72, metadata !37), !dbg !78
  %0 = load i64, i64* @a, align 8, !dbg !79, !tbaa !40
  tail call void @llvm.dbg.value(metadata %struct.B* %d, metadata !73, metadata !37), !dbg !80
  %call = call %struct.B* @_ZN1BC1El(%struct.B* %d, i64 %0), !dbg !80
  call void @llvm.dbg.value(metadata i8 1, metadata !72, metadata !81), !dbg !78
  call void @llvm.dbg.value(metadata i8 1, metadata !72, metadata !82), !dbg !78
  call void @llvm.dbg.value(metadata i8 1, metadata !56, metadata !81), !dbg !75
  call void @llvm.dbg.value(metadata i8 1, metadata !56, metadata !82), !dbg !75
  call void @llvm.dbg.declare(metadata %struct.A* undef, metadata !56, metadata !37), !dbg !75
  %1 = getelementptr inbounds %struct.A, %struct.A* %agg.tmp.i.i, i64 0, i32 0, !dbg !83
  call void @llvm.lifetime.start(i64 24, i8* %1), !dbg !83
  %2 = load i64, i64* @a, align 8, !dbg !85, !tbaa !40
  %call.i.i5 = invoke noalias i8* @_Znwm(i64 %2) #4
          to label %call.i.i.noexc unwind label %lpad, !dbg !86

call.i.i.noexc:                                   ; preds = %entry
  %agg.tmp.sroa.4.17..sroa_idx = getelementptr inbounds [7 x i8], [7 x i8]* %agg.tmp.sroa.4, i64 0, i64 0, !dbg !87
  %agg.tmp.sroa.2.1..sroa_idx = getelementptr inbounds [15 x i8], [15 x i8]* %agg.tmp.sroa.2, i64 0, i64 0, !dbg !87
  store i8* %call.i.i5, i8** bitcast (i32** @b to i8**), align 8, !dbg !88, !tbaa !46
  store i8 1, i8* %1, align 8, !dbg !89
  %agg.tmp.sroa.2.0..sroa_raw_idx = getelementptr inbounds i8, i8* %1, i64 1, !dbg !89
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %agg.tmp.sroa.2.0..sroa_raw_idx, i8* %agg.tmp.sroa.2.1..sroa_idx, i64 15, i32 1, i1 false), !dbg !89
  %agg.tmp.sroa.3.0..sroa_idx = getelementptr inbounds %struct.A, %struct.A* %agg.tmp.i.i, i64 0, i32 2, !dbg !89
  store i8 1, i8* %agg.tmp.sroa.3.0..sroa_idx, align 8, !dbg !89
  %agg.tmp.sroa.4.0..sroa_raw_idx = getelementptr inbounds i8, i8* %1, i64 17, !dbg !89
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %agg.tmp.sroa.4.0..sroa_raw_idx, i8* %agg.tmp.sroa.4.17..sroa_idx, i64 7, i32 1, i1 false), !dbg !89
  invoke void @_Z2f91A(%struct.A* %agg.tmp.i.i)
          to label %invoke.cont unwind label %lpad, !dbg !90

invoke.cont:                                      ; preds = %call.i.i.noexc
  call void @llvm.lifetime.end(i64 24, i8* %1), !dbg !91
  call void @llvm.dbg.value(metadata %struct.B* %d, metadata !73, metadata !37), !dbg !80
  %call1 = call %struct.B* @_ZN1BD1Ev(%struct.B* %d) #3, !dbg !92
  ret void, !dbg !92

lpad:                                             ; preds = %call.i.i.noexc, %entry
  %3 = landingpad { i8*, i32 }
          cleanup, !dbg !92
  call void @llvm.dbg.value(metadata %struct.B* %d, metadata !73, metadata !37), !dbg !80
  %call2 = call %struct.B* @_ZN1BD1Ev(%struct.B* %d) #3, !dbg !92
  resume { i8*, i32 } %3, !dbg !92
}

declare %struct.B* @_ZN1BC1El(%struct.B*, i64)

declare i32 @__gxx_personality_v0(...)

; Function Attrs: nounwind
declare %struct.B* @_ZN1BD1Ev(%struct.B*) #3

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, metadata, metadata) #0

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start(i64, i8* nocapture) #2

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end(i64, i8* nocapture) #2

attributes #0 = { nounwind readnone }
attributes #1 = { nobuiltin }
attributes #2 = { argmemonly nounwind }
attributes #3 = { nounwind }
attributes #4 = { builtin }

!llvm.dbg.cu = !{!8}
!llvm.module.flags = !{!29, !30}
!llvm.ident = !{!31}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = !DIGlobalVariable(name: "a", scope: null, file: !2, line: 1, type: !3, isLocal: false, isDefinition: true)
!2 = !DIFile(filename: "test.cpp", directory: "")
!3 = !DIBasicType(name: "long int", size: 64, align: 64, encoding: DW_ATE_signed)
!4 = !DIGlobalVariableExpression(var: !5, expr: !DIExpression())
!5 = !DIGlobalVariable(name: "b", scope: null, file: !2, line: 7, type: !6, isLocal: false, isDefinition: true)
!6 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !7, size: 64, align: 64)
!7 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!8 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !9, producer: "clang version 3.7.0 ", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !10, retainedTypes: !11, globals: !28, imports: !10)
!9 = !DIFile(filename: "<stdin>", directory: "")
!10 = !{}
!11 = !{!12, !6, !19}
!12 = !DICompositeType(tag: DW_TAG_structure_type, name: "A", file: !2, line: 2, size: 192, align: 64, elements: !13, identifier: "_ZTS1A")
!13 = !{!14, !16, !18}
!14 = !DIDerivedType(tag: DW_TAG_member, name: "x4", scope: !12, file: !2, line: 3, baseType: !15, size: 8, align: 8)
!15 = !DIBasicType(name: "bool", size: 8, align: 8, encoding: DW_ATE_boolean)
!16 = !DIDerivedType(tag: DW_TAG_member, name: "x5", scope: !12, file: !2, line: 4, baseType: !17, size: 64, align: 64, offset: 64)
!17 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64, align: 64)
!18 = !DIDerivedType(tag: DW_TAG_member, name: "x6", scope: !12, file: !2, line: 5, baseType: !15, size: 8, align: 8, offset: 128)
!19 = !DICompositeType(tag: DW_TAG_structure_type, name: "B", file: !2, line: 8, size: 8, align: 8, elements: !20, identifier: "_ZTS1B")
!20 = !{!21, !25}
!21 = !DISubprogram(name: "B", scope: !19, file: !2, line: 9, type: !22, isLocal: false, isDefinition: false, scopeLine: 9, flags: DIFlagPrototyped, isOptimized: true)
!22 = !DISubroutineType(types: !23)
!23 = !{null, !24, !3}
!24 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !19, size: 64, align: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!25 = !DISubprogram(name: "~B", scope: !19, file: !2, line: 10, type: !26, isLocal: false, isDefinition: false, scopeLine: 10, flags: DIFlagPrototyped, isOptimized: true)
!26 = !DISubroutineType(types: !27)
!27 = !{null, !24}
!28 = !{!0, !4}
!29 = !{i32 2, !"Dwarf Version", i32 2}
!30 = !{i32 2, !"Debug Info Version", i32 3}
!31 = !{!"clang version 3.7.0 "}
!32 = distinct !DISubprogram(name: "f13", linkageName: "_Z3f131A", scope: !2, file: !2, line: 13, type: !33, isLocal: false, isDefinition: true, scopeLine: 13, flags: DIFlagPrototyped, isOptimized: true, unit: !8, variables: !35)
!33 = !DISubroutineType(types: !34)
!34 = !{null, !12}
!35 = !{!36}
!36 = !DILocalVariable(name: "p1", arg: 1, scope: !32, file: !2, line: 13, type: !12)
!37 = !DIExpression(DW_OP_deref)
!38 = !DILocation(line: 13, column: 12, scope: !32)
!39 = !DILocation(line: 14, column: 37, scope: !32)
!40 = !{!41, !41, i64 0}
!41 = !{!"long", !42, i64 0}
!42 = !{!"omnipotent char", !43, i64 0}
!43 = !{!"Simple C/C++ TBAA"}
!44 = !DILocation(line: 14, column: 14, scope: !32)
!45 = !DILocation(line: 14, column: 5, scope: !32)
!46 = !{!47, !47, i64 0}
!47 = !{!"any pointer", !42, i64 0}
!48 = !DILocation(line: 15, column: 6, scope: !32)
!49 = !{i64 0, i64 1, !50, i64 8, i64 8, !46, i64 16, i64 1, !50}
!50 = !{!51, !51, i64 0}
!51 = !{!"bool", !42, i64 0}
!52 = !DILocation(line: 15, column: 3, scope: !32)
!53 = !DILocation(line: 16, column: 1, scope: !32)
!54 = distinct !DISubprogram(name: "f11", linkageName: "_Z3f111A", scope: !2, file: !2, line: 17, type: !33, isLocal: false, isDefinition: true, scopeLine: 17, flags: DIFlagPrototyped, isOptimized: true, unit: !8, variables: !55)
!55 = !{!56}
!56 = !DILocalVariable(name: "p1", arg: 1, scope: !54, file: !2, line: 17, type: !12)
!57 = !DILocation(line: 17, column: 12, scope: !54)
!58 = !DILocation(line: 17, column: 22, scope: !54)
!59 = !DILocation(line: 13, column: 12, scope: !32, inlinedAt: !60)
!60 = distinct !DILocation(line: 17, column: 18, scope: !54)
!61 = !DILocation(line: 14, column: 37, scope: !32, inlinedAt: !60)
!62 = !DILocation(line: 14, column: 14, scope: !32, inlinedAt: !60)
!63 = !DILocation(line: 14, column: 5, scope: !32, inlinedAt: !60)
!64 = !DILocation(line: 15, column: 6, scope: !32, inlinedAt: !60)
!65 = !DILocation(line: 15, column: 3, scope: !32, inlinedAt: !60)
!66 = !DILocation(line: 16, column: 1, scope: !32, inlinedAt: !60)
!67 = !DILocation(line: 17, column: 27, scope: !54)
!68 = distinct !DISubprogram(name: "f16", linkageName: "_Z3f16v", scope: !2, file: !2, line: 18, type: !69, isLocal: false, isDefinition: true, scopeLine: 18, flags: DIFlagPrototyped, isOptimized: true, unit: !8, variables: !71)
!69 = !DISubroutineType(types: !70)
!70 = !{null}
!71 = !{!72, !73}
!72 = !DILocalVariable(name: "c", scope: !68, file: !2, line: 19, type: !12)
!73 = !DILocalVariable(name: "d", scope: !68, file: !2, line: 20, type: !19)
!74 = !DIExpression(DW_OP_LLVM_fragment, 8, 120)
!75 = !DILocation(line: 17, column: 12, scope: !54, inlinedAt: !76)
!76 = distinct !DILocation(line: 22, column: 3, scope: !68)
!77 = !DIExpression(DW_OP_LLVM_fragment, 136, 56)
!78 = !DILocation(line: 19, column: 5, scope: !68)
!79 = !DILocation(line: 20, column: 7, scope: !68)
!80 = !DILocation(line: 20, column: 5, scope: !68)
!81 = !DIExpression(DW_OP_LLVM_fragment, 0, 8)
!82 = !DIExpression(DW_OP_LLVM_fragment, 128, 8)
!83 = !DILocation(line: 13, column: 12, scope: !32, inlinedAt: !84)
!84 = distinct !DILocation(line: 17, column: 18, scope: !54, inlinedAt: !76)
!85 = !DILocation(line: 14, column: 37, scope: !32, inlinedAt: !84)
!86 = !DILocation(line: 14, column: 14, scope: !32, inlinedAt: !84)
!87 = !DILocation(line: 22, column: 7, scope: !68)
!88 = !DILocation(line: 14, column: 5, scope: !32, inlinedAt: !84)
!89 = !DILocation(line: 15, column: 6, scope: !32, inlinedAt: !84)
!90 = !DILocation(line: 15, column: 3, scope: !32, inlinedAt: !84)
!91 = !DILocation(line: 16, column: 1, scope: !32, inlinedAt: !84)
!92 = !DILocation(line: 23, column: 1, scope: !68)

