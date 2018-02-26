; RUN: opt -sroa -S -o - %s | FileCheck %s
; Generated from clang -c  -O2 -g -target x86_64-pc-windows-msvc
; struct A { double x1[]; };
; struct x2 {
;   x2(int) : x3() {}
;   int x3;
; };
; int x4();
; x2 x5() {
;   x2 a(x4());
;   return a;
; }
; void *operator new(size_t, void *);
; struct B {
;   B() { new (x8.x1) x2(x5()); }
;   A x8;
; };
; void x9() { B(); }
source_filename = "t.cpp"
target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.11.0"

%struct.B = type { %struct.A }
%struct.A = type { [0 x double], [8 x i8] }

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #0

declare i32 @"\01?x4@@YAHXZ"() local_unnamed_addr

define void @"\01?x9@@YAXXZ"() local_unnamed_addr !dbg !8 {
entry:
  %agg.tmp.ensured = alloca %struct.B, align 8
  call void @llvm.dbg.declare(metadata %struct.B* %agg.tmp.ensured, metadata !11, metadata !DIExpression()), !dbg !24
  %call.i.i = call i32 @"\01?x4@@YAHXZ"(), !dbg !46, !noalias !47
  %x3.i.i.i = bitcast %struct.B* %agg.tmp.ensured to i32*, !dbg !50
  store i32 0, i32* %x3.i.i.i, align 4, !dbg !50, !tbaa !57, !alias.scope !47
  ; CHECK: call void @llvm.dbg.value(metadata i32 0, metadata ![[A:.*]], metadata !DIExpression())
  ; CHECK: ![[A]] = !DILocalVariable(name: "a",
  ret void, !dbg !62
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #0

attributes #0 = { nounwind readnone speculatable }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !6}
!llvm.ident = !{!7}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 6.0.0 (trunk 319058) (llvm/trunk 319066)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "t.cpp", directory: "/")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 2}
!6 = !{i32 7, !"PIC Level", i32 2}
!7 = !{!"clang version 6.0.0 (trunk 319058) (llvm/trunk 319066)"}
!8 = distinct !DISubprogram(name: "x9", linkageName: "\01?x9@@YAXXZ", scope: !1, file: !1, line: 16, type: !9, isLocal: false, isDefinition: true, scopeLine: 16, flags: DIFlagPrototyped, isOptimized: true, unit: !0, variables: !2)
!9 = !DISubroutineType(types: !10)
!10 = !{null}
!11 = !DILocalVariable(name: "a", scope: !12, file: !1, line: 8, type: !15)
!12 = distinct !DISubprogram(name: "x5", linkageName: "\01?x5@@YA?AUx2@@XZ", scope: !1, file: !1, line: 7, type: !13, isLocal: false, isDefinition: true, scopeLine: 7, flags: DIFlagPrototyped, isOptimized: true, unit: !0, variables: !23)
!13 = !DISubroutineType(types: !14)
!14 = !{!15}
!15 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "x2", file: !1, line: 2, size: 32, elements: !16, identifier: ".?AUx2@@")
!16 = !{!17, !19}
!17 = !DIDerivedType(tag: DW_TAG_member, name: "x3", scope: !15, file: !1, line: 4, baseType: !18, size: 32)
!18 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!19 = !DISubprogram(name: "x2", scope: !15, file: !1, line: 3, type: !20, isLocal: false, isDefinition: false, scopeLine: 3, flags: DIFlagPrototyped, isOptimized: true)
!20 = !DISubroutineType(types: !21)
!21 = !{null, !22, !18}
!22 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !15, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!23 = !{!11}
!24 = !DILocation(line: 8, column: 6, scope: !12, inlinedAt: !25)
!25 = distinct !DILocation(line: 13, column: 24, scope: !26, inlinedAt: !45)
!26 = distinct !DILexicalBlock(scope: !27, file: !1, line: 13, column: 7)
!27 = distinct !DISubprogram(name: "B", linkageName: "\01??0B@@QEAA@XZ", scope: !28, file: !1, line: 13, type: !39, isLocal: false, isDefinition: true, scopeLine: 13, flags: DIFlagPrototyped, isOptimized: true, unit: !0, declaration: !38, variables: !42)
!28 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "B", file: !1, line: 12, size: 64, elements: !29, identifier: ".?AUB@@")
!29 = !{!30, !38}
!30 = !DIDerivedType(tag: DW_TAG_member, name: "x8", scope: !28, file: !1, line: 14, baseType: !31, size: 64)
!31 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "A", file: !1, line: 1, size: 64, elements: !32, identifier: ".?AUA@@")
!32 = !{!33}
!33 = !DIDerivedType(tag: DW_TAG_member, name: "x1", scope: !31, file: !1, line: 1, baseType: !34)
!34 = !DICompositeType(tag: DW_TAG_array_type, baseType: !35, elements: !36)
!35 = !DIBasicType(name: "double", size: 64, encoding: DW_ATE_float)
!36 = !{!37}
!37 = !DISubrange(count: -1)
!38 = !DISubprogram(name: "B", scope: !28, file: !1, line: 13, type: !39, isLocal: false, isDefinition: false, scopeLine: 13, flags: DIFlagPrototyped, isOptimized: true)
!39 = !DISubroutineType(types: !40)
!40 = !{null, !41}
!41 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !28, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!42 = !{!43}
!43 = !DILocalVariable(name: "this", arg: 1, scope: !27, type: !44, flags: DIFlagArtificial | DIFlagObjectPointer)
!44 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !28, size: 64)
!45 = distinct !DILocation(line: 16, column: 13, scope: !8)
!46 = !DILocation(line: 8, column: 8, scope: !12, inlinedAt: !25)
!47 = !{!48}
!48 = distinct !{!48, !49, !"\01?x5@@YA?AUx2@@XZ: %agg.result"}
!49 = distinct !{!49, !"\01?x5@@YA?AUx2@@XZ"}
!50 = !DILocation(line: 3, column: 13, scope: !51, inlinedAt: !56)
!51 = distinct !DISubprogram(name: "x2", linkageName: "\01??0x2@@QEAA@H@Z", scope: !15, file: !1, line: 3, type: !20, isLocal: false, isDefinition: true, scopeLine: 3, flags: DIFlagPrototyped, isOptimized: true, unit: !0, declaration: !19, variables: !52)
!52 = !{!53, !54}
!53 = !DILocalVariable(arg: 2, scope: !51, file: !1, line: 3, type: !18)
!54 = !DILocalVariable(name: "this", arg: 1, scope: !51, type: !55, flags: DIFlagArtificial | DIFlagObjectPointer)
!55 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !15, size: 64)
!56 = distinct !DILocation(line: 8, column: 6, scope: !12, inlinedAt: !25)
!57 = !{!58, !59, i64 0}
!58 = !{!"?AUx2@@", !59, i64 0}
!59 = !{!"int", !60, i64 0}
!60 = !{!"omnipotent char", !61, i64 0}
!61 = !{!"Simple C++ TBAA"}
!62 = !DILocation(line: 16, column: 18, scope: !8)
