; struct A {
;   A();
;   virtual ~A();
; };
; struct B : A {
;   B();
;   virtual ~B();
; };
; B::B() {}
; CHECK: __ZN1BC1Ev:
; CHECK:     .loc	1 [[@LINE-2]] 0 prologue_end
; CHECK-NOT: .loc	1 0 0 prologue_end

; The location of the prologue_end marker should not be affected by the presence
; of CFI instructions.

; RUN: llc -O0 -filetype=asm -mtriple=thumbv7-apple-ios < %s | FileCheck %s
; RUN: llc -O0 -filetype=asm -mtriple=thumbv6-apple-ios < %s | FileCheck %s

; ModuleID = 'test1.cpp'
target datalayout = "e-m:o-p:32:32-f64:32:64-v64:32:64-v128:32:128-a:0:32-n32-S32"
target triple = "thumbv7-apple-ios"

%struct.B = type { %struct.A }
%struct.A = type { i32 (...)** }

@_ZTV1B = external unnamed_addr constant [4 x i8*]

; Function Attrs: nounwind
define %struct.B* @_ZN1BC2Ev(%struct.B* %this) unnamed_addr #0 align 2 !dbg !28 {
entry:
  tail call void @llvm.dbg.value(metadata %struct.B* %this, i64 0, metadata !30, metadata !40), !dbg !41
  %0 = getelementptr inbounds %struct.B, %struct.B* %this, i32 0, i32 0, !dbg !42
  %call = tail call %struct.A* @_ZN1AC2Ev(%struct.A* %0) #3, !dbg !42
  %1 = getelementptr inbounds %struct.B, %struct.B* %this, i32 0, i32 0, i32 0, !dbg !42
  store i32 (...)** bitcast (i8** getelementptr inbounds ([4 x i8*], [4 x i8*]* @_ZTV1B, i32 0, i32 2) to i32 (...)**), i32 (...)*** %1, align 4, !dbg !42, !tbaa !43
  ret %struct.B* %this, !dbg !42
}

declare %struct.A* @_ZN1AC2Ev(%struct.A*)

; Function Attrs: nounwind
define %struct.B* @_ZN1BC1Ev(%struct.B* %this) unnamed_addr #0 align 2 !dbg !32 {
entry:
  tail call void @llvm.dbg.value(metadata %struct.B* %this, i64 0, metadata !34, metadata !40), !dbg !46
  tail call void @llvm.dbg.value(metadata %struct.B* %this, i64 0, metadata !47, metadata !40) #3, !dbg !49
  %0 = getelementptr inbounds %struct.B, %struct.B* %this, i32 0, i32 0, !dbg !50
  %call.i = tail call %struct.A* @_ZN1AC2Ev(%struct.A* %0) #3, !dbg !50
  %1 = getelementptr inbounds %struct.B, %struct.B* %this, i32 0, i32 0, i32 0, !dbg !50
  store i32 (...)** bitcast (i8** getelementptr inbounds ([4 x i8*], [4 x i8*]* @_ZTV1B, i32 0, i32 2) to i32 (...)**), i32 (...)*** %1, align 4, !dbg !50, !tbaa !43
  ret %struct.B* %this, !dbg !48
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #2

attributes #0 = { nounwind }
attributes #2 = { nounwind readnone }
attributes #3 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!35, !36, !37, !38}
!llvm.ident = !{!39}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.6.0 (trunk 224279) (llvm/trunk 224283)", isOptimized: true, emissionKind: FullDebug, file: !1, enums: !2, retainedTypes: !3, globals: !2, imports: !2)
!1 = !DIFile(filename: "<stdin>", directory: "")
!2 = !{}
!3 = !{!4, !13}
!4 = !DICompositeType(tag: DW_TAG_structure_type, name: "B", line: 5, size: 32, align: 32, file: !5, elements: !6, vtableHolder: !13, identifier: "_ZTS1B")
!5 = !DIFile(filename: "test1.cpp", directory: "")
!6 = !{!7, !8, !12}
!7 = !DIDerivedType(tag: DW_TAG_inheritance, scope: !4, baseType: !13)
!8 = !DISubprogram(name: "B", line: 6, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true, scopeLine: 6, file: !5, scope: !4, type: !9)
!9 = !DISubroutineType(types: !10)
!10 = !{null, !11}
!11 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 32, align: 32, flags: DIFlagArtificial | DIFlagObjectPointer, baseType: !4)
!12 = !DISubprogram(name: "~B", line: 7, isLocal: false, isDefinition: false, virtuality: DW_VIRTUALITY_virtual, flags: DIFlagPrototyped, isOptimized: true, scopeLine: 7, file: !5, scope: !4, type: !9, containingType: !4)
!13 = !DICompositeType(tag: DW_TAG_structure_type, name: "A", line: 1, size: 32, align: 32, file: !5, elements: !14, vtableHolder: !13, identifier: "_ZTS1A")
!14 = !{!15, !22, !26}
!15 = !DIDerivedType(tag: DW_TAG_member, name: "_vptr$A", size: 32, flags: DIFlagArtificial, file: !5, scope: !16, baseType: !17)
!16 = !DIFile(filename: "test1.cpp", directory: "")
!17 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 32, baseType: !18)
!18 = !DIDerivedType(tag: DW_TAG_pointer_type, name: "__vtbl_ptr_type", size: 32, baseType: !19)
!19 = !DISubroutineType(types: !20)
!20 = !{!21}
!21 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!22 = !DISubprogram(name: "A", line: 2, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: true, scopeLine: 2, file: !5, scope: !13, type: !23)
!23 = !DISubroutineType(types: !24)
!24 = !{null, !25}
!25 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 32, align: 32, flags: DIFlagArtificial | DIFlagObjectPointer, baseType: !13)
!26 = !DISubprogram(name: "~A", line: 3, isLocal: false, isDefinition: false, virtuality: DW_VIRTUALITY_virtual, flags: DIFlagPrototyped, isOptimized: true, scopeLine: 3, file: !5, scope: !13, type: !23, containingType: !13)
!28 = distinct !DISubprogram(name: "B", linkageName: "_ZN1BC2Ev", line: 9, isLocal: false, isDefinition: true, flags: DIFlagPrototyped, isOptimized: true, unit: !0, scopeLine: 9, file: !5, scope: !4, type: !9, declaration: !8, variables: !29)
!29 = !{!30}
!30 = !DILocalVariable(name: "this", arg: 1, flags: DIFlagArtificial | DIFlagObjectPointer, scope: !28, type: !31)
!31 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 32, align: 32, baseType: !4)
!32 = distinct !DISubprogram(name: "B", linkageName: "_ZN1BC1Ev", line: 9, isLocal: false, isDefinition: true, flags: DIFlagPrototyped, isOptimized: true, unit: !0, scopeLine: 9, file: !5, scope: !4, type: !9, declaration: !8, variables: !33)
!33 = !{!34}
!34 = !DILocalVariable(name: "this", arg: 1, flags: DIFlagArtificial | DIFlagObjectPointer, scope: !32, type: !31)
!35 = !{i32 2, !"Dwarf Version", i32 4}
!36 = !{i32 2, !"Debug Info Version", i32 3}
!37 = !{i32 1, !"wchar_size", i32 4}
!38 = !{i32 1, !"min_enum_size", i32 4}
!39 = !{!"clang version 3.6.0 (trunk 224279) (llvm/trunk 224283)"}
!40 = !DIExpression()
!41 = !DILocation(line: 0, scope: !28)
!42 = !DILocation(line: 9, scope: !28)
!43 = !{!44, !44, i64 0}
!44 = !{!"vtable pointer", !45, i64 0}
!45 = !{!"Simple C/C++ TBAA"}
!46 = !DILocation(line: 0, scope: !32)
!47 = !DILocalVariable(name: "this", arg: 1, flags: DIFlagArtificial | DIFlagObjectPointer, scope: !28, type: !31)
!48 = !DILocation(line: 9, scope: !32)
!49 = !DILocation(line: 0, scope: !28, inlinedAt: !48)
!50 = !DILocation(line: 9, scope: !28, inlinedAt: !48)
