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
; CHECK: define

; Function Attrs: inlinehint nounwind ssp uwtable
define linkonce_odr void @_ZN1BC1Ev(%struct.B* %this) unnamed_addr #1 align 2 !dbg !25 {
entry:
  %this.addr = alloca %struct.B*, align 8
  store %struct.B* %this, %struct.B** %this.addr, align 8
  ; CHECK-NOT: @llvm.dbg.declare
  call void @llvm.dbg.declare(metadata %struct.B** %this.addr, metadata !30, metadata !32), !dbg !33
  %this1 = load %struct.B*, %struct.B** %this.addr, align 8
  call void @_ZN1BC2Ev(%struct.B* %this1) #2, !dbg !34
  ret void, !dbg !34
  ; CHECK: call void @_ZN1BC2Ev(%struct.B* %this1){{.*}} !dbg !
}

attributes #0 = { nounwind readnone }
attributes #1 = { inlinehint nounwind ssp uwtable }
attributes #2 = { nounwind }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!21, !22, !23}
!llvm.ident = !{!24}

; CHECK-NEG-NOT: !DI{{Basic|Composite|Derived}}Type

!0 = distinct !DIGlobalVariableExpression(var: !1)
!1 = !DIGlobalVariable(name: "b", scope: !2, file: !3, line: 5, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "clang version 4.0.0 (trunk 282583) (llvm/trunk 282611)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5)
!3 = !DIFile(filename: "t.cpp", directory: "/")
!4 = !{}
!5 = !{!0}
!6 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "B", file: !3, line: 4, size: 64, align: 64, elements: !7, vtableHolder: !9, identifier: "_ZTS1B")
!7 = !{!8}
!8 = !DIDerivedType(tag: DW_TAG_inheritance, scope: !6, baseType: !9)
!9 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "A", file: !3, line: 1, size: 64, align: 64, elements: !10, vtableHolder: !9, identifier: "_ZTS1A")
!10 = !{!11, !17}
!11 = !DIDerivedType(tag: DW_TAG_member, name: "_vptr$A", scope: !3, file: !3, baseType: !12, size: 64, flags: DIFlagArtificial)
!12 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !13, size: 64)
!13 = !DIDerivedType(tag: DW_TAG_pointer_type, name: "__vtbl_ptr_type", baseType: !14, size: 64)
!14 = !DISubroutineType(types: !15)
!15 = !{!16}
!16 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!17 = !DISubprogram(name: "~A", scope: !9, file: !3, line: 2, type: !18, isLocal: false, isDefinition: false, scopeLine: 2, containingType: !9, virtuality: DW_VIRTUALITY_virtual, virtualIndex: 0, flags: DIFlagPrototyped, isOptimized: false)
!18 = !DISubroutineType(types: !19)
!19 = !{null, !20}
!20 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !9, size: 64, align: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!21 = !{i32 2, !"Dwarf Version", i32 4}
!22 = !{i32 2, !"Debug Info Version", i32 3}
!23 = !{i32 1, !"PIC Level", i32 2}
!24 = !{!"clang version 4.0.0 (trunk 282583) (llvm/trunk 282611)"}
; Only referenced by the type system.
; CHECK-NEG-NOT: !DISubprogram(name: "~A"
!25 = distinct !DISubprogram(name: "B", linkageName: "_ZN1BC1Ev", scope: !6, file: !3, line: 4, type: !26, isLocal: false, isDefinition: true, scopeLine: 4, flags: DIFlagArtificial | DIFlagPrototyped, isOptimized: false, unit: !2, declaration: !29, variables: !4)
!26 = !DISubroutineType(types: !27)
!27 = !{null, !28}
!28 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !6, size: 64, align: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!29 = !DISubprogram(name: "B", scope: !6, type: !26, isLocal: false, isDefinition: false, flags: DIFlagArtificial | DIFlagPrototyped, isOptimized: false)
!30 = !DILocalVariable(name: "this", arg: 1, scope: !25, type: !31, flags: DIFlagArtificial | DIFlagObjectPointer)
!31 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !6, size: 64, align: 64)
!32 = !DIExpression()
!33 = !DILocation(line: 0, scope: !25)
!34 = !DILocation(line: 4, column: 8, scope: !25)

; CHECK: !DISubprogram(name: "B", scope: ![[FILE:.*]], file: ![[FILE]],
; CHECK-NOT: containingType:
; CHECK-NEG-NOT: !DISubprogram(name: "B", {{.*}}, isDefinition: false
