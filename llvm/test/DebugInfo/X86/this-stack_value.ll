; RUN: llc -filetype=asm -o - %s | FileCheck %s --check-prefix=ASM
; RUN: llc -filetype=obj -o - %s | llvm-dwarfdump --debug-dump=info - | FileCheck %s
;
; Generated at -O2 from:
;   struct B;
;   class A {
;     int a1 = 23;
;     int a2 = 42;
;   };
;   struct B {
;     A a;
;     int b = 48;
;   };
;    
;   B *getB() { return new B(); }
;
; The inlined A::this pointer has the same location as B::this, but it may not be
; modified by the debugger.
;
; ASM: [DW_OP_stack_value]
; CHECK:  DW_AT_location {{.*}} 70 00 9f
;                               rax+0, stack-value
source_filename = "ab.cpp"
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.12.0"

%struct.B = type { %class.A, i32 }
%class.A = type { i32, i32 }

; Function Attrs: ssp uwtable
define noalias nonnull %struct.B* @_Z4getBv() local_unnamed_addr #0 !dbg !7 {
entry:
  %call = tail call i8* @_Znwm(i64 12) #3, !dbg !20
  %0 = bitcast i8* %call to %struct.B*, !dbg !20
  tail call void @llvm.dbg.value(metadata %struct.B* %0, i64 0, metadata !21, metadata !28), !dbg !29
  tail call void @llvm.dbg.value(metadata %struct.B* %0, i64 0, metadata !31, metadata !28), !dbg !34
  tail call void @llvm.dbg.value(metadata %struct.B* %0, i64 0, metadata !36, metadata !44), !dbg !45
  tail call void @llvm.dbg.value(metadata %struct.B* %0, i64 0, metadata !47, metadata !44), !dbg !50
  %a1.i.i.i.i = bitcast i8* %call to i32*, !dbg !52
  store i32 23, i32* %a1.i.i.i.i, align 4, !dbg !52, !tbaa !53
  %a2.i.i.i.i = getelementptr inbounds i8, i8* %call, i64 4, !dbg !58
  %1 = bitcast i8* %a2.i.i.i.i to i32*, !dbg !58
  store i32 42, i32* %1, align 4, !dbg !58, !tbaa !59
  %b.i.i = getelementptr inbounds i8, i8* %call, i64 8, !dbg !60
  %2 = bitcast i8* %b.i.i to i32*, !dbg !60
  store i32 48, i32* %2, align 4, !dbg !60, !tbaa !61
  ret %struct.B* %0, !dbg !63
}

declare noalias nonnull i8* @_Znwm(i64) local_unnamed_addr #1
declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #2

attributes #0 = { ssp uwtable }
attributes #1 = { nobuiltin }
attributes #2 = { nounwind readnone }
attributes #3 = { builtin }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 5.0.0 (trunk 301093) (llvm/trunk 301093)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "ab.cpp", directory: "/")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"PIC Level", i32 2}
!7 = distinct !DISubprogram(name: "getB", linkageName: "_Z4getBv", scope: !1, file: !1, line: 11, type: !8, isLocal: false, isDefinition: true, scopeLine: 11, flags: DIFlagPrototyped, isOptimized: true, unit: !0, variables: !2)
!8 = !DISubroutineType(types: !9)
!9 = !{!10}
!10 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !11, size: 64)
!11 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "B", file: !1, line: 6, size: 96, elements: !12, identifier: "_ZTS1B")
!12 = !{!13, !19}
!13 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !11, file: !1, line: 7, baseType: !14, size: 64)
!14 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "A", file: !1, line: 2, size: 64, elements: !15, identifier: "_ZTS1A")
!15 = !{!16, !18}
!16 = !DIDerivedType(tag: DW_TAG_member, name: "a1", scope: !14, file: !1, line: 3, baseType: !17, size: 32)
!17 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!18 = !DIDerivedType(tag: DW_TAG_member, name: "a2", scope: !14, file: !1, line: 4, baseType: !17, size: 32, offset: 32)
!19 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !11, file: !1, line: 8, baseType: !17, size: 32, offset: 64)
!20 = !DILocation(line: 11, column: 20, scope: !7)
!21 = !DILocalVariable(name: "this", arg: 1, scope: !22, type: !10, flags: DIFlagArtificial | DIFlagObjectPointer)
!22 = distinct !DISubprogram(name: "B", linkageName: "_ZN1BC1Ev", scope: !11, file: !1, line: 6, type: !23, isLocal: false, isDefinition: true, scopeLine: 6, flags: DIFlagArtificial | DIFlagPrototyped, isOptimized: true, unit: !0, declaration: !26, variables: !27)
!23 = !DISubroutineType(types: !24)
!24 = !{null, !25}
!25 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !11, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!26 = !DISubprogram(name: "B", scope: !11, type: !23, isLocal: false, isDefinition: false, flags: DIFlagArtificial | DIFlagPrototyped, isOptimized: true)
!27 = !{!21}
!28 = !DIExpression()
!29 = !DILocation(line: 0, scope: !22, inlinedAt: !30)
!30 = distinct !DILocation(line: 11, column: 24, scope: !7)
!31 = !DILocalVariable(name: "this", arg: 1, scope: !32, type: !10, flags: DIFlagArtificial | DIFlagObjectPointer)
!32 = distinct !DISubprogram(name: "B", linkageName: "_ZN1BC2Ev", scope: !11, file: !1, line: 6, type: !23, isLocal: false, isDefinition: true, scopeLine: 6, flags: DIFlagArtificial | DIFlagPrototyped, isOptimized: true, unit: !0, declaration: !26, variables: !33)
!33 = !{!31}
!34 = !DILocation(line: 0, scope: !32, inlinedAt: !35)
!35 = distinct !DILocation(line: 6, column: 8, scope: !22, inlinedAt: !30)
!36 = !DILocalVariable(name: "this", arg: 1, scope: !37, type: !43, flags: DIFlagArtificial | DIFlagObjectPointer)
!37 = distinct !DISubprogram(name: "A", linkageName: "_ZN1AC1Ev", scope: !14, file: !1, line: 2, type: !38, isLocal: false, isDefinition: true, scopeLine: 2, flags: DIFlagArtificial | DIFlagPrototyped, isOptimized: true, unit: !0, declaration: !41, variables: !42)
!38 = !DISubroutineType(types: !39)
!39 = !{null, !40}
!40 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !14, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!41 = !DISubprogram(name: "A", scope: !14, type: !38, isLocal: false, isDefinition: false, flags: DIFlagPublic | DIFlagArtificial | DIFlagPrototyped, isOptimized: true)
!42 = !{!36}
!43 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !14, size: 64)
!44 = !DIExpression(DW_OP_stack_value)
!45 = !DILocation(line: 0, scope: !37, inlinedAt: !46)
!46 = distinct !DILocation(line: 6, column: 8, scope: !32, inlinedAt: !35)
!47 = !DILocalVariable(name: "this", arg: 1, scope: !48, type: !43, flags: DIFlagArtificial | DIFlagObjectPointer)
!48 = distinct !DISubprogram(name: "A", linkageName: "_ZN1AC2Ev", scope: !14, file: !1, line: 2, type: !38, isLocal: false, isDefinition: true, scopeLine: 2, flags: DIFlagArtificial | DIFlagPrototyped, isOptimized: true, unit: !0, declaration: !41, variables: !49)
!49 = !{!47}
!50 = !DILocation(line: 0, scope: !48, inlinedAt: !51)
!51 = distinct !DILocation(line: 2, column: 7, scope: !37, inlinedAt: !46)
!52 = !DILocation(line: 3, column: 7, scope: !48, inlinedAt: !51)
!53 = !{!54, !55, i64 0}
!54 = !{!"_ZTS1A", !55, i64 0, !55, i64 4}
!55 = !{!"int", !56, i64 0}
!56 = !{!"omnipotent char", !57, i64 0}
!57 = !{!"Simple C++ TBAA"}
!58 = !DILocation(line: 4, column: 7, scope: !48, inlinedAt: !51)
!59 = !{!54, !55, i64 4}
!60 = !DILocation(line: 8, column: 7, scope: !32, inlinedAt: !35)
!61 = !{!62, !55, i64 8}
!62 = !{!"_ZTS1B", !54, i64 0, !55, i64 8}
!63 = !DILocation(line: 11, column: 13, scope: !7)
