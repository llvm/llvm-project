; RUN: llc -O0 -filetype=obj < %s | llvm-dwarfdump - | FileCheck %s
; Test that a variable with multiple entries in the MMI table makes it into the
; debug info.
;
; CHECK: DW_TAG_inlined_subroutine
; CHECK:    "_Z3f111A"
; CHECK: DW_TAG_formal_parameter
; CHECK: DW_AT_location [DW_FORM_block1]    (<0x0b> 91 51 9d 78 08 91 4a 9d 38 88 01 )
;  -- fbreg -47, bit-piece 120 8 , fbreg -54, bit-piece 56 136 ------^
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
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-apple-ios"

%struct.A = type { i8, i8*, i8 }
%struct.B = type { i8 }

@a = global i64 0, align 8
@b = global i32* null, align 8

define void @_Z3f131A(%struct.A* nocapture readonly %p1) #0 !dbg !25 {
entry:
  %agg.tmp = alloca %struct.A, align 8
  tail call void @llvm.dbg.declare(metadata %struct.A* %p1, metadata !30, metadata !46), !dbg !47
  %0 = load i64, i64* @a, align 8, !dbg !48, !tbaa !49
  %call = tail call noalias i8* @_Znwm(i64 %0) #5, !dbg !53
  store i8* %call, i8** bitcast (i32** @b to i8**), align 8, !dbg !54, !tbaa !55
  %1 = getelementptr inbounds %struct.A, %struct.A* %agg.tmp, i64 0, i32 0, !dbg !57
  %2 = getelementptr inbounds %struct.A, %struct.A* %p1, i64 0, i32 0, !dbg !57
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %1, i8* %2, i64 24, i32 8, i1 false), !dbg !57, !tbaa.struct !58
  call void @_Z2f91A(%struct.A* %agg.tmp), !dbg !61
  ret void, !dbg !62
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: nobuiltin
declare noalias i8* @_Znwm(i64) #2

declare void @_Z2f91A(%struct.A*) #0

; Function Attrs: nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture readonly, i64, i32, i1) #3

define void @_Z3f111A(%struct.A* nocapture readonly %p1) #0 !dbg !31 {
entry:
  %agg.tmp.i = alloca %struct.A, align 8
  tail call void @llvm.dbg.declare(metadata %struct.A* %p1, metadata !33, metadata !46), !dbg !63
  %0 = getelementptr inbounds %struct.A, %struct.A* %p1, i64 0, i32 0, !dbg !64
  %1 = getelementptr inbounds %struct.A, %struct.A* %agg.tmp.i, i64 0, i32 0, !dbg !65
  call void @llvm.lifetime.start(i64 24, i8* %1), !dbg !65
  %2 = load i64, i64* @a, align 8, !dbg !67, !tbaa !49
  %call.i = tail call noalias i8* @_Znwm(i64 %2) #5, !dbg !68
  store i8* %call.i, i8** bitcast (i32** @b to i8**), align 8, !dbg !69, !tbaa !55
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %1, i8* %0, i64 24, i32 8, i1 false), !dbg !70
  call void @_Z2f91A(%struct.A* %agg.tmp.i), !dbg !71
  call void @llvm.lifetime.end(i64 24, i8* %1), !dbg !72
  ret void, !dbg !73
}

define void @_Z3f16v() #0 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) !dbg !34 {
entry:
  %agg.tmp.i.i = alloca %struct.A, align 8
  %d = alloca %struct.B, align 1
  %agg.tmp.sroa.2 = alloca [15 x i8], align 1
  %agg.tmp.sroa.4 = alloca [7 x i8], align 1
  tail call void @llvm.dbg.declare(metadata [15 x i8]* %agg.tmp.sroa.2, metadata !74, metadata !76), !dbg !77
  tail call void @llvm.dbg.declare(metadata [7 x i8]* %agg.tmp.sroa.4, metadata !74, metadata !78), !dbg !77
  tail call void @llvm.dbg.declare(metadata %struct.A* undef, metadata !38, metadata !79), !dbg !80
  %0 = load i64, i64* @a, align 8, !dbg !81, !tbaa !49
  tail call void @llvm.dbg.value(metadata %struct.B* %d, i64 0, metadata !39, metadata !79), !dbg !82
  %call = call %struct.B* @_ZN1BC1El(%struct.B* %d, i64 %0), !dbg !82
  call void @llvm.dbg.value(metadata i8 1, i64 0, metadata !38, metadata !83), !dbg !80
  call void @llvm.dbg.value(metadata i8 1, i64 0, metadata !38, metadata !84), !dbg !80
  call void @llvm.dbg.value(metadata i8 1, i64 0, metadata !74, metadata !83), !dbg !77
  call void @llvm.dbg.value(metadata i8 1, i64 0, metadata !74, metadata !84), !dbg !77
  call void @llvm.dbg.declare(metadata %struct.A* undef, metadata !74, metadata !46), !dbg !77
  %1 = getelementptr inbounds %struct.A, %struct.A* %agg.tmp.i.i, i64 0, i32 0, !dbg !85
  call void @llvm.lifetime.start(i64 24, i8* %1), !dbg !85
  %2 = load i64, i64* @a, align 8, !dbg !87, !tbaa !49
  %call.i.i5 = invoke noalias i8* @_Znwm(i64 %2) #5
          to label %call.i.i.noexc unwind label %lpad, !dbg !88

call.i.i.noexc:                                   ; preds = %entry
  %agg.tmp.sroa.4.17..sroa_idx = getelementptr inbounds [7 x i8], [7 x i8]* %agg.tmp.sroa.4, i64 0, i64 0, !dbg !89
  %agg.tmp.sroa.2.1..sroa_idx = getelementptr inbounds [15 x i8], [15 x i8]* %agg.tmp.sroa.2, i64 0, i64 0, !dbg !89
  store i8* %call.i.i5, i8** bitcast (i32** @b to i8**), align 8, !dbg !90, !tbaa !55
  store i8 1, i8* %1, align 8, !dbg !91
  %agg.tmp.sroa.2.0..sroa_raw_idx = getelementptr inbounds i8, i8* %1, i64 1, !dbg !91
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %agg.tmp.sroa.2.0..sroa_raw_idx, i8* %agg.tmp.sroa.2.1..sroa_idx, i64 15, i32 1, i1 false), !dbg !91
  %agg.tmp.sroa.3.0..sroa_idx = getelementptr inbounds %struct.A, %struct.A* %agg.tmp.i.i, i64 0, i32 2, !dbg !91
  store i8 1, i8* %agg.tmp.sroa.3.0..sroa_idx, align 8, !dbg !91
  %agg.tmp.sroa.4.0..sroa_raw_idx = getelementptr inbounds i8, i8* %1, i64 17, !dbg !91
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %agg.tmp.sroa.4.0..sroa_raw_idx, i8* %agg.tmp.sroa.4.17..sroa_idx, i64 7, i32 1, i1 false), !dbg !91
  invoke void @_Z2f91A(%struct.A* %agg.tmp.i.i)
          to label %invoke.cont unwind label %lpad, !dbg !92

invoke.cont:                                      ; preds = %call.i.i.noexc
  call void @llvm.lifetime.end(i64 24, i8* %1), !dbg !93
  call void @llvm.dbg.value(metadata %struct.B* %d, i64 0, metadata !39, metadata !79), !dbg !82
  %call1 = call %struct.B* @_ZN1BD1Ev(%struct.B* %d) #3, !dbg !94
  ret void, !dbg !94

lpad:                                             ; preds = %call.i.i.noexc, %entry
  %3 = landingpad { i8*, i32 }
          cleanup, !dbg !94
  call void @llvm.dbg.value(metadata %struct.B* %d, i64 0, metadata !39, metadata !79), !dbg !82
  %call2 = call %struct.B* @_ZN1BD1Ev(%struct.B* %d) #3, !dbg !94
  resume { i8*, i32 } %3, !dbg !94
}

declare %struct.B* @_ZN1BC1El(%struct.B*, i64)

declare i32 @__gxx_personality_v0(...)

; Function Attrs: nounwind
declare %struct.B* @_ZN1BD1Ev(%struct.B*) #4

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #1

; Function Attrs: nounwind
declare void @llvm.lifetime.start(i64, i8* nocapture) #3

; Function Attrs: nounwind
declare void @llvm.lifetime.end(i64, i8* nocapture) #3

attributes #1 = { nounwind readnone }
attributes #2 = { nobuiltin }
attributes #3 = { nounwind }
attributes #4 = { nounwind  }
attributes #5 = { builtin }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!43, !44}
!llvm.ident = !{!45}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.7.0 ", isOptimized: true, emissionKind: 1, file: !1, enums: !2, retainedTypes: !3, subprograms: !24, globals: !40, imports: !2)
!1 = !DIFile(filename: "<stdin>", directory: "")
!2 = !{}
!3 = !{!4, !12, !14}
!4 = !DICompositeType(tag: DW_TAG_structure_type, name: "A", line: 2, size: 192, align: 64, file: !5, elements: !6, identifier: "_ZTS1A")
!5 = !DIFile(filename: "test.cpp", directory: "")
!6 = !{!7, !9, !11}
!7 = !DIDerivedType(tag: DW_TAG_member, name: "x4", line: 3, size: 8, align: 8, file: !5, scope: !"_ZTS1A", baseType: !8)
!8 = !DIBasicType(tag: DW_TAG_base_type, name: "bool", size: 8, align: 8, encoding: DW_ATE_boolean)
!9 = !DIDerivedType(tag: DW_TAG_member, name: "x5", line: 4, size: 64, align: 64, offset: 64, file: !5, scope: !"_ZTS1A", baseType: !10)
!10 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, baseType: null)
!11 = !DIDerivedType(tag: DW_TAG_member, name: "x6", line: 5, size: 8, align: 8, offset: 128, file: !5, scope: !"_ZTS1A", baseType: !8)
!12 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, baseType: !13)
!13 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!14 = !DICompositeType(tag: DW_TAG_structure_type, name: "B", line: 8, size: 8, align: 8, file: !5, elements: !15, identifier: "_ZTS1B")
!15 = !{!16, !21}
!16 = !DISubprogram(name: "B", line: 9, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true, scopeLine: 9, file: !5, scope: !"_ZTS1B", type: !17)
!17 = !DISubroutineType(types: !18)
!18 = !{null, !19, !20}
!19 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, flags: DIFlagArtificial | DIFlagObjectPointer, baseType: !"_ZTS1B")
!20 = !DIBasicType(tag: DW_TAG_base_type, name: "long int", size: 64, align: 64, encoding: DW_ATE_signed)
!21 = !DISubprogram(name: "~B", line: 10, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true, scopeLine: 10, file: !5, scope: !"_ZTS1B", type: !22)
!22 = !DISubroutineType(types: !23)
!23 = !{null, !19}
!24 = !{!25, !31, !34}
!25 = distinct !DISubprogram(name: "f13", linkageName: "_Z3f131A", line: 13, isLocal: false, isDefinition: true, flags: DIFlagPrototyped, isOptimized: true, scopeLine: 13, file: !5, scope: !26, type: !27, variables: !29)
!26 = !DIFile(filename: "test.cpp", directory: "")
!27 = !DISubroutineType(types: !28)
!28 = !{null, !"_ZTS1A"}
!29 = !{!30}
!30 = !DILocalVariable(name: "p1", line: 13, arg: 1, scope: !25, file: !26, type: !"_ZTS1A")
!31 = distinct !DISubprogram(name: "f11", linkageName: "_Z3f111A", line: 17, isLocal: false, isDefinition: true, flags: DIFlagPrototyped, isOptimized: true, scopeLine: 17, file: !5, scope: !26, type: !27, variables: !32)
!32 = !{!33}
!33 = !DILocalVariable(name: "p1", line: 17, arg: 1, scope: !31, file: !26, type: !"_ZTS1A")
!34 = distinct !DISubprogram(name: "f16", linkageName: "_Z3f16v", line: 18, isLocal: false, isDefinition: true, flags: DIFlagPrototyped, isOptimized: true, scopeLine: 18, file: !5, scope: !26, type: !35, variables: !37)
!35 = !DISubroutineType(types: !36)
!36 = !{null}
!37 = !{!38, !39}
!38 = !DILocalVariable(name: "c", line: 19, scope: !34, file: !26, type: !"_ZTS1A")
!39 = !DILocalVariable(name: "d", line: 20, scope: !34, file: !26, type: !"_ZTS1B")
!40 = !{!41, !42}
!41 = !DIGlobalVariable(name: "a", line: 1, isLocal: false, isDefinition: true, scope: null, file: !26, type: !20, variable: i64* @a)
!42 = !DIGlobalVariable(name: "b", line: 7, isLocal: false, isDefinition: true, scope: null, file: !26, type: !12, variable: i32** @b)
!43 = !{i32 2, !"Dwarf Version", i32 2}
!44 = !{i32 2, !"Debug Info Version", i32 3}
!45 = !{!"clang version 3.7.0 "}
!46 = !DIExpression(DW_OP_deref)
!47 = !DILocation(line: 13, column: 12, scope: !25)
!48 = !DILocation(line: 14, column: 37, scope: !25)
!49 = !{!50, !50, i64 0}
!50 = !{!"long", !51, i64 0}
!51 = !{!"omnipotent char", !52, i64 0}
!52 = !{!"Simple C/C++ TBAA"}
!53 = !DILocation(line: 14, column: 14, scope: !25)
!54 = !DILocation(line: 14, column: 5, scope: !25)
!55 = !{!56, !56, i64 0}
!56 = !{!"any pointer", !51, i64 0}
!57 = !DILocation(line: 15, column: 6, scope: !25)
!58 = !{i64 0, i64 1, !59, i64 8, i64 8, !55, i64 16, i64 1, !59}
!59 = !{!60, !60, i64 0}
!60 = !{!"bool", !51, i64 0}
!61 = !DILocation(line: 15, column: 3, scope: !25)
!62 = !DILocation(line: 16, column: 1, scope: !25)
!63 = !DILocation(line: 17, column: 12, scope: !31)
!64 = !DILocation(line: 17, column: 22, scope: !31)
!65 = !DILocation(line: 13, column: 12, scope: !25, inlinedAt: !66)
!66 = distinct !DILocation(line: 17, column: 18, scope: !31)
!67 = !DILocation(line: 14, column: 37, scope: !25, inlinedAt: !66)
!68 = !DILocation(line: 14, column: 14, scope: !25, inlinedAt: !66)
!69 = !DILocation(line: 14, column: 5, scope: !25, inlinedAt: !66)
!70 = !DILocation(line: 15, column: 6, scope: !25, inlinedAt: !66)
!71 = !DILocation(line: 15, column: 3, scope: !25, inlinedAt: !66)
!72 = !DILocation(line: 16, column: 1, scope: !25, inlinedAt: !66)
!73 = !DILocation(line: 17, column: 27, scope: !31)
!74 = !DILocalVariable(name: "p1", line: 17, arg: 1, scope: !31, file: !26, type: !"_ZTS1A")
!75 = distinct !DILocation(line: 22, column: 3, scope: !34)
!76 = !DIExpression(DW_OP_bit_piece, 8, 120)
!77 = !DILocation(line: 17, column: 12, scope: !31, inlinedAt: !75)
!78 = !DIExpression(DW_OP_bit_piece, 136, 56)
!79 = !DIExpression(DW_OP_deref)
!80 = !DILocation(line: 19, column: 5, scope: !34)
!81 = !DILocation(line: 20, column: 7, scope: !34)
!82 = !DILocation(line: 20, column: 5, scope: !34)
!83 = !DIExpression(DW_OP_bit_piece, 0, 8)
!84 = !DIExpression(DW_OP_bit_piece, 128, 8)
!85 = !DILocation(line: 13, column: 12, scope: !25, inlinedAt: !86)
!86 = distinct !DILocation(line: 17, column: 18, scope: !31, inlinedAt: !75)
!87 = !DILocation(line: 14, column: 37, scope: !25, inlinedAt: !86)
!88 = !DILocation(line: 14, column: 14, scope: !25, inlinedAt: !86)
!89 = !DILocation(line: 22, column: 7, scope: !34)
!90 = !DILocation(line: 14, column: 5, scope: !25, inlinedAt: !86)
!91 = !DILocation(line: 15, column: 6, scope: !25, inlinedAt: !86)
!92 = !DILocation(line: 15, column: 3, scope: !25, inlinedAt: !86)
!93 = !DILocation(line: 16, column: 1, scope: !25, inlinedAt: !86)
!94 = !DILocation(line: 23, column: 1, scope: !34)
