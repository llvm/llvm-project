; REQUIRES: object-emission

; RUN: %llc_dwarf -debugger-tune=gdb -filetype=obj -o %t.o < %s
; RUN: llvm-dwarfdump -debug-pubnames %t.o | FileCheck %s
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

source_filename = "test/DebugInfo/Generic/dwarf-public-names.ll"

%struct.C = type { i8 }

@_ZN1C22static_member_variableE = global i32 0, align 4, !dbg !0
@global_variable = global %struct.C zeroinitializer, align 1, !dbg !15
@_ZN2ns25global_namespace_variableE = global i32 1, align 4, !dbg !17

; Function Attrs: nounwind uwtable
define void @_ZN1C15member_functionEv(%struct.C* %this) #0 align 2 !dbg !23 {
entry:
  %this.addr = alloca %struct.C*, align 8
  store %struct.C* %this, %struct.C** %this.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.C** %this.addr, metadata !24, metadata !26), !dbg !27
  %this1 = load %struct.C*, %struct.C** %this.addr
  store i32 0, i32* @_ZN1C22static_member_variableE, align 4, !dbg !28
  ret void, !dbg !29
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: nounwind uwtable
define i32 @_ZN1C22static_member_functionEv() #0 align 2 !dbg !30 {
entry:
  %0 = load i32, i32* @_ZN1C22static_member_variableE, align 4, !dbg !31
  ret i32 %0, !dbg !31
}

; Function Attrs: nounwind uwtable
define i32 @_Z15global_functionv() #0 !dbg !32 {
entry:
  ret i32 -1, !dbg !33
}

; Function Attrs: nounwind uwtable
define void @_ZN2ns25global_namespace_functionEv() #0 !dbg !34 {
entry:
  call void @_ZN1C15member_functionEv(%struct.C* @global_variable), !dbg !37
  ret void, !dbg !38
}

attributes #0 = { nounwind uwtable }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!20}
!llvm.module.flags = !{!22}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = !DIGlobalVariable(name: "static_member_variable", linkageName: "_ZN1C22static_member_variableE", scope: !2, file: !3, line: 7, type: !6, isLocal: false, isDefinition: true, declaration: !5)
!2 = !DICompositeType(tag: DW_TAG_structure_type, name: "C", file: !3, line: 1, size: 8, align: 8, elements: !4)
!3 = !DIFile(filename: "dwarf-public-names.cpp", directory: "/usr2/kparzysz/s.hex/t")
!4 = !{!5, !7, !12}
!5 = !DIDerivedType(tag: DW_TAG_member, name: "static_member_variable", scope: !2, file: !3, line: 4, baseType: !6, flags: DIFlagStaticMember)
!6 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!7 = !DISubprogram(name: "member_function", linkageName: "_ZN1C15member_functionEv", scope: !2, file: !3, line: 2, type: !8, isLocal: false, isDefinition: false, scopeLine: 2, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, retainedNodes: !11)
!8 = !DISubroutineType(types: !9)
!9 = !{null, !10}
!10 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !2, size: 64, align: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!11 = !{}
!12 = !DISubprogram(name: "static_member_function", linkageName: "_ZN1C22static_member_functionEv", scope: !2, file: !3, line: 3, type: !13, isLocal: false, isDefinition: false, scopeLine: 3, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, retainedNodes: !11)
!13 = !DISubroutineType(types: !14)
!14 = !{!6}
!15 = !DIGlobalVariableExpression(var: !16, expr: !DIExpression())
!16 = !DIGlobalVariable(name: "global_variable", scope: null, file: !3, line: 17, type: !2, isLocal: false, isDefinition: true) ; previously: invalid DW_TAG_base_type
!17 = !DIGlobalVariableExpression(var: !18, expr: !DIExpression())
!18 = !DIGlobalVariable(name: "global_namespace_variable", linkageName: "_ZN2ns25global_namespace_variableE", scope: !19, file: !3, line: 27, type: !6, isLocal: false, isDefinition: true)
!19 = !DINamespace(name: "ns", scope: null)
!20 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "clang version 3.3 (http://llvm.org/git/clang.git a09cd8103a6a719cb2628cdf0c91682250a17bd2) (http://llvm.org/git/llvm.git 47d03cec0afca0c01ae42b82916d1d731716cd20)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !11, retainedTypes: !11, globals: !21, imports: !11) ; previously: invalid DW_TAG_base_type
!21 = !{!0, !15, !17}
!22 = !{i32 1, !"Debug Info Version", i32 3}
!23 = distinct !DISubprogram(name: "member_function", linkageName: "_ZN1C15member_functionEv", scope: null, file: !3, line: 9, type: !8, isLocal: false, isDefinition: true, scopeLine: 9, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !20, declaration: !7, retainedNodes: !11)
!24 = !DILocalVariable(name: "this", arg: 1, scope: !23, file: !3, line: 9, type: !25, flags: DIFlagArtificial | DIFlagObjectPointer)
!25 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !2, size: 64, align: 64)
!26 = !DIExpression()
!27 = !DILocation(line: 9, scope: !23)
!28 = !DILocation(line: 10, scope: !23)
!29 = !DILocation(line: 11, scope: !23)
!30 = distinct !DISubprogram(name: "static_member_function", linkageName: "_ZN1C22static_member_functionEv", scope: null, file: !3, line: 13, type: !13, isLocal: false, isDefinition: true, scopeLine: 13, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !20, declaration: !12, retainedNodes: !11)
!31 = !DILocation(line: 14, scope: !30)
!32 = distinct !DISubprogram(name: "global_function", linkageName: "_Z15global_functionv", scope: !3, file: !3, line: 19, type: !13, isLocal: false, isDefinition: true, scopeLine: 19, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !20, retainedNodes: !11)
!33 = !DILocation(line: 20, scope: !32)
!34 = distinct !DISubprogram(name: "global_namespace_function", linkageName: "_ZN2ns25global_namespace_functionEv", scope: !19, file: !3, line: 24, type: !35, isLocal: false, isDefinition: true, scopeLine: 24, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !20, retainedNodes: !11)
!35 = !DISubroutineType(types: !36)
!36 = !{null}
!37 = !DILocation(line: 25, scope: !34)
!38 = !DILocation(line: 26, scope: !34)

