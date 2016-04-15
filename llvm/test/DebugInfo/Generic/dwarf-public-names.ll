; REQUIRES: object-emission

; RUN: %llc_dwarf -generate-dwarf-pub-sections=Enable -filetype=obj -o %t.o < %s
; RUN: llvm-dwarfdump -debug-dump=pubnames %t.o | FileCheck %s
; ModuleID = 'dwarf-public-names.cpp'
;
; Generated from:
;
; struct C {
;   void member_function();
;   static int static_member_function();
;   static int static_member_variable;
; };
;
; int C::static_member_variable = 0;
;
; void C::member_function() {
;   static_member_variable = 0;
; }
;
; int C::static_member_function() {
;   return static_member_variable;
; }
;
; C global_variable;
;
; int global_function() {
;   return -1;
; }
;
; namespace ns {
;   void global_namespace_function() {
;     global_variable.member_function();
;   }
;   int global_namespace_variable = 1;
; }

; Skip the output to the header of the pubnames section.
; CHECK: debug_pubnames
; CHECK: version = 0x0002

; Check for each name in the output.
; CHECK-DAG: "ns"
; CHECK-DAG: "C::static_member_function"
; CHECK-DAG: "global_variable"
; CHECK-DAG: "ns::global_namespace_variable"
; CHECK-DAG: "ns::global_namespace_function"
; CHECK-DAG: "global_function"
; CHECK-DAG: "C::static_member_variable"
; CHECK-DAG: "C::member_function"

%struct.C = type { i8 }

@_ZN1C22static_member_variableE = global i32 0, align 4
@global_variable = global %struct.C zeroinitializer, align 1
@_ZN2ns25global_namespace_variableE = global i32 1, align 4

define void @_ZN1C15member_functionEv(%struct.C* %this) nounwind uwtable align 2 !dbg !3 {
entry:
  %this.addr = alloca %struct.C*, align 8
  store %struct.C* %this, %struct.C** %this.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.C** %this.addr, metadata !28, metadata !DIExpression()), !dbg !30
  %this1 = load %struct.C*, %struct.C** %this.addr
  store i32 0, i32* @_ZN1C22static_member_variableE, align 4, !dbg !31
  ret void, !dbg !32
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

define i32 @_ZN1C22static_member_functionEv() nounwind uwtable align 2 !dbg !18 {
entry:
  %0 = load i32, i32* @_ZN1C22static_member_variableE, align 4, !dbg !33
  ret i32 %0, !dbg !33
}

define i32 @_Z15global_functionv() nounwind uwtable !dbg !19 {
entry:
  ret i32 -1, !dbg !34
}

define void @_ZN2ns25global_namespace_functionEv() nounwind uwtable !dbg !20 {
entry:
  call void @_ZN1C15member_functionEv(%struct.C* @global_variable), !dbg !35
  ret void, !dbg !36
}

attributes #0 = { nounwind uwtable }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!38}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.3 (http://llvm.org/git/clang.git a09cd8103a6a719cb2628cdf0c91682250a17bd2) (http://llvm.org/git/llvm.git 47d03cec0afca0c01ae42b82916d1d731716cd20)", isOptimized: false, emissionKind: FullDebug, file: !37, enums: !1, retainedTypes: !1, globals: !24, imports:  !1)
!1 = !{}
!3 = distinct !DISubprogram(name: "member_function", linkageName: "_ZN1C15member_functionEv", line: 9, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 9, file: !4, scope: null, type: !5, declaration: !12, variables: !1)
!4 = !DIFile(filename: "dwarf-public-names.cpp", directory: "/usr2/kparzysz/s.hex/t")
!5 = !DISubroutineType(types: !6)
!6 = !{null, !7}
!7 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, flags: DIFlagArtificial | DIFlagObjectPointer, baseType: !8)
!8 = !DICompositeType(tag: DW_TAG_structure_type, name: "C", line: 1, size: 8, align: 8, file: !37, elements: !9)
!9 = !{!10, !12, !14}
!10 = !DIDerivedType(tag: DW_TAG_member, name: "static_member_variable", line: 4, flags: DIFlagStaticMember, file: !37, scope: !8, baseType: !11)
!11 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!12 = !DISubprogram(name: "member_function", linkageName: "_ZN1C15member_functionEv", line: 2, isLocal: false, isDefinition: false, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, scopeLine: 2, file: !4, scope: !8, type: !5, variables: !13)
!13 = !{} ; previously: invalid DW_TAG_base_type
!14 = !DISubprogram(name: "static_member_function", linkageName: "_ZN1C22static_member_functionEv", line: 3, isLocal: false, isDefinition: false, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, scopeLine: 3, file: !4, scope: !8, type: !15, variables: !17)
!15 = !DISubroutineType(types: !16)
!16 = !{!11}
!17 = !{} ; previously: invalid DW_TAG_base_type
!18 = distinct !DISubprogram(name: "static_member_function", linkageName: "_ZN1C22static_member_functionEv", line: 13, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 13, file: !4, scope: null, type: !15, declaration: !14, variables: !1)
!19 = distinct !DISubprogram(name: "global_function", linkageName: "_Z15global_functionv", line: 19, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 19, file: !4, scope: !4, type: !15, variables: !1)
!20 = distinct !DISubprogram(name: "global_namespace_function", linkageName: "_ZN2ns25global_namespace_functionEv", line: 24, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 24, file: !4, scope: !21, type: !22, variables: !1)
!21 = !DINamespace(name: "ns", line: 23, file: !4, scope: null)
!22 = !DISubroutineType(types: !23)
!23 = !{null}
!24 = !{!25, !26, !27}
!25 = !DIGlobalVariable(name: "static_member_variable", linkageName: "_ZN1C22static_member_variableE", line: 7, isLocal: false, isDefinition: true, scope: !8, file: !4, type: !11, variable: i32* @_ZN1C22static_member_variableE, declaration: !10)
!26 = !DIGlobalVariable(name: "global_variable", line: 17, isLocal: false, isDefinition: true, scope: null, file: !4, type: !8, variable: %struct.C* @global_variable)
!27 = !DIGlobalVariable(name: "global_namespace_variable", linkageName: "_ZN2ns25global_namespace_variableE", line: 27, isLocal: false, isDefinition: true, scope: !21, file: !4, type: !11, variable: i32* @_ZN2ns25global_namespace_variableE)
!28 = !DILocalVariable(name: "this", line: 9, arg: 1, flags: DIFlagArtificial | DIFlagObjectPointer, scope: !3, file: !4, type: !29)
!29 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, baseType: !8)
!30 = !DILocation(line: 9, scope: !3)
!31 = !DILocation(line: 10, scope: !3)
!32 = !DILocation(line: 11, scope: !3)
!33 = !DILocation(line: 14, scope: !18)
!34 = !DILocation(line: 20, scope: !19)
!35 = !DILocation(line: 25, scope: !20)
!36 = !DILocation(line: 26, scope: !20)
!37 = !DIFile(filename: "dwarf-public-names.cpp", directory: "/usr2/kparzysz/s.hex/t")
!38 = !{i32 1, !"Debug Info Version", i32 3}
