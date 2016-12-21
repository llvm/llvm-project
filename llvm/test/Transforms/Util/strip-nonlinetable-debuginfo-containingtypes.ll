; RUN: opt -S -strip-nonlinetable-debuginfo %s -o %t.ll
; RUN: cat %t.ll | FileCheck %s
; RUN: cat %t.ll | FileCheck %s --check-prefix=CHECK-NEG
;
; This test provides coverage for setting the containing type of a DISubprogram.
;
; Generated an reduced from:
; struct A {
;   virtual ~A();
; };
; struct B : A {};
; B b;

source_filename = "t.cpp"
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.12.0"

%struct.B = type { %struct.A }
%struct.A = type { i32 (...)** }

; CHECK: @b = global
; CHECK-NOT: !dbg
@b = global %struct.B zeroinitializer, align 8, !dbg !0

declare void @_ZN1BC2Ev(%struct.B*) unnamed_addr

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #0

; Function Attrs: inlinehint nounwind ssp uwtable
; CHECK: define
define linkonce_odr void @_ZN1BC1Ev(%struct.B* %this) unnamed_addr #1 align 2 !dbg !24 {
entry:
  %this.addr = alloca %struct.B*, align 8
  store %struct.B* %this, %struct.B** %this.addr, align 8
  ; CHECK-NOT: @llvm.dbg.declare
  call void @llvm.dbg.declare(metadata %struct.B** %this.addr, metadata !29, metadata !31), !dbg !32
  %this1 = load %struct.B*, %struct.B** %this.addr, align 8
  ; CHECK: call void @_ZN1BC2Ev(%struct.B* %this1){{.*}} !dbg !
  call void @_ZN1BC2Ev(%struct.B* %this1) #2, !dbg !33
  ret void, !dbg !33
}

attributes #0 = { nounwind readnone }
attributes #1 = { inlinehint nounwind ssp uwtable }
attributes #2 = { nounwind }

!llvm.dbg.cu = !{!1}
!llvm.module.flags = !{!20, !21, !22}
!llvm.ident = !{!23}

; CHECK-NEG-NOT: !DI{{Basic|Composite|Derived}}Type

!0 = distinct !DIGlobalVariableExpression(var: !DIGlobalVariable(name: "b", scope: !1, file: !2, line: 5, type: !5, isLocal: false, isDefinition: true))
!1 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !2, producer: "clang version 4.0.0 (trunk 282583) (llvm/trunk 282611)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !3, globals: !4)
!2 = !DIFile(filename: "t.cpp", directory: "/")
!3 = !{}
!4 = !{!0}
!5 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "B", file: !2, line: 4, size: 64, align: 64, elements: !6, vtableHolder: !8, identifier: "_ZTS1B")
!6 = !{!7}
!7 = !DIDerivedType(tag: DW_TAG_inheritance, scope: !5, baseType: !8)
!8 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "A", file: !2, line: 1, size: 64, align: 64, elements: !9, vtableHolder: !8, identifier: "_ZTS1A")
!9 = !{!10, !16}
!10 = !DIDerivedType(tag: DW_TAG_member, name: "_vptr$A", scope: !2, file: !2, baseType: !11, size: 64, flags: DIFlagArtificial)
!11 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !12, size: 64)
!12 = !DIDerivedType(tag: DW_TAG_pointer_type, name: "__vtbl_ptr_type", baseType: !13, size: 64)
!13 = !DISubroutineType(types: !14)
!14 = !{!15}
!15 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
; Only referenced by the type system.
; CHECK-NEG-NOT: !DISubprogram(name: "~A"
!16 = !DISubprogram(name: "~A", scope: !8, file: !2, line: 2, type: !17, isLocal: false, isDefinition: false, scopeLine: 2, containingType: !8, virtuality: DW_VIRTUALITY_virtual, virtualIndex: 0, flags: DIFlagPrototyped, isOptimized: false)
!17 = !DISubroutineType(types: !18)
!18 = !{null, !19}
!19 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !8, size: 64, align: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!20 = !{i32 2, !"Dwarf Version", i32 4}
!21 = !{i32 2, !"Debug Info Version", i32 3}
!22 = !{i32 1, !"PIC Level", i32 2}
!23 = !{!"clang version 4.0.0 (trunk 282583) (llvm/trunk 282611)"}
; CHECK: !DISubprogram(name: "B", scope: ![[FILE:.*]], file: ![[FILE]],
; CHECK-NOT: containingType:
!24 = distinct !DISubprogram(name: "B", linkageName: "_ZN1BC1Ev", scope: !5, file: !2, line: 4, type: !25, isLocal: false, isDefinition: true, scopeLine: 4, flags: DIFlagArtificial | DIFlagPrototyped, isOptimized: false, unit: !1, declaration: !28, variables: !3)
!25 = !DISubroutineType(types: !26)
!26 = !{null, !27}
!27 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !5, size: 64, align: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
; CHECK-NEG-NOT: !DISubprogram(name: "B", {{.*}}, isDefinition: false
!28 = !DISubprogram(name: "B", scope: !5, type: !25, isLocal: false, isDefinition: false, flags: DIFlagArtificial | DIFlagPrototyped, isOptimized: false)
!29 = !DILocalVariable(name: "this", arg: 1, scope: !24, type: !30, flags: DIFlagArtificial | DIFlagObjectPointer)
!30 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !5, size: 64, align: 64)
!31 = !DIExpression()
!32 = !DILocation(line: 0, scope: !24)
!33 = !DILocation(line: 4, column: 8, scope: !24)
