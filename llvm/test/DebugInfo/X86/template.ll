
; RUN: llc -mtriple=x86_64-linux -O0 -filetype=obj < %s | llvm-dwarfdump -v -debug-info - | FileCheck %s
; RUN: llc -mtriple=x86_64-linux -O0 -filetype=obj < %s | not llvm-dwarfdump -verify - | FileCheck %s --check-prefix VERIFY

; IR generated with `clang++ -g -emit-llvm -S` from the following code:
; template<int x, int*, template<typename> class y, decltype(nullptr) n, int ...z>  int func() {
;  var<int> = 5;
;  return var<int>;
; }
; template<typename> struct y_impl { struct nested { }; };
; int glbl = func<3, &glbl, y_impl, nullptr, 1, 2>();
; y_impl<int>::nested n;

; VERIFY-NOT: error: DIE has DW_AT_type with incompatible tag DW_TAG_unspecified_type
; VERIFY: error: DIEs have overlapping address ranges

; CHECK: [[INT:0x[0-9a-f]*]]:{{ *}}DW_TAG_base_type
; CHECK-NEXT: DW_AT_name{{.*}} = "int"

; CHECK: DW_TAG_structure_type
; CHECK: DW_AT_name{{.*}}"y_impl<int>"
; CHECK-NOT: {{TAG|NULL}}
; CHECK: DW_TAG_template_type_parameter

; CHECK: DW_TAG_variable
; CHECK-NEXT: DW_AT_name{{.*}}"var"
; CHECK-NOT: NULL
; CHECK: DW_TAG_template_type_parameter
; CHECK-NEXT: DW_AT_type{{.*}}=> {[[INT]]}
; CHECK-NEXT: DW_AT_name{{.*}}= "T"


; CHECK: DW_AT_name{{.*}}"func<3, &glbl, y_impl, nullptr, 1, 2>"
; CHECK-NOT: NULL
; CHECK: DW_TAG_template_value_parameter
; CHECK-NEXT: DW_AT_type{{.*}}=> {[[INT]]}
; CHECK-NEXT: DW_AT_name{{.*}}= "x"
; CHECK-NEXT: DW_AT_const_value [DW_FORM_sdata]{{.*}}(3)

; CHECK: DW_TAG_template_value_parameter
; CHECK-NEXT: DW_AT_type{{.*}}=> {[[INTPTR:0x[0-9a-f]*]]}

; The address of the global 'glbl', followed by DW_OP_stack_value (9f), to use
; the value immediately, rather than indirecting through the address.

; CHECK-NEXT: DW_AT_location [DW_FORM_exprloc]{{ *}}(DW_OP_addr 0x0, DW_OP_stack_value)
; CHECK-NOT: NULL

; CHECK: DW_TAG_GNU_template_template_param
; CHECK-NEXT: DW_AT_name{{.*}}= "y"
; CHECK-NEXT: DW_AT_GNU_template_name{{.*}}= "y_impl"
; CHECK-NOT: NULL

; CHECK: DW_TAG_template_value_parameter
; CHECK-NEXT: DW_AT_type{{.*}}=> {[[NULLPTR:0x[0-9a-f]*]]}
; CHECK-NEXT: DW_AT_name{{.*}}= "n"
; CHECK-NEXT: DW_AT_const_value [DW_FORM_udata]{{.*}}(0)

; CHECK: DW_TAG_GNU_template_parameter_pack
; CHECK-NOT: NULL
; CHECK: DW_TAG_template_value_parameter
; CHECK-NEXT: DW_AT_type{{.*}}=> {[[INT]]}
; CHECK-NEXT: DW_AT_const_value  [DW_FORM_sdata]{{.*}}(1)
; CHECK-NOT: NULL
; CHECK: DW_TAG_template_value_parameter
; CHECK-NEXT: DW_AT_type{{.*}}=> {[[INT]]}
; CHECK-NEXT: DW_AT_const_value  [DW_FORM_sdata]{{.*}}(2)

; CHECK: [[INTPTR]]:{{ *}}DW_TAG_pointer_type
; CHECK-NEXT: DW_AT_type{{.*}} => {[[INT]]}

; CHECK: [[NULLPTR]]:{{ *}}DW_TAG_unspecified_type
; CHECK-NEXT: DW_AT_name{{.*}}= "decltype(nullptr)"

source_filename = "test/DebugInfo/X86/template.ll"
%"struct.y_impl<int>::nested" = type { i8 }

@glbl = dso_local global i32 0, align 4, !dbg !0
@_Z3varIiE = linkonce_odr dso_local global i32 0, align 4, !dbg !13
@n = dso_local global %"struct.y_impl<int>::nested" zeroinitializer, align 1, !dbg !6
@llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @_GLOBAL__sub_I_upstream_test.cpp, i8* null }]

; Function Attrs: noinline uwtable
define internal void @__cxx_global_var_init() #0 section ".text.startup" !dbg !21 {
entry:
  %call = call i32 @_Z4funcILi3EXadL_Z4glblEE6y_implLDn0EJLi1ELi2EEEiv(), !dbg !24
  store i32 %call, i32* @glbl, align 4, !dbg !24
  ret void, !dbg !24
}

; Function Attrs: noinline nounwind optnone uwtable
define linkonce_odr dso_local i32 @_Z4funcILi3EXadL_Z4glblEE6y_implLDn0EJLi1ELi2EEEiv() #1 !dbg !25 {
entry:
  store i32 5, i32* @_Z3varIiE, align 4, !dbg !39
  %0 = load i32, i32* @_Z3varIiE, align 4, !dbg !40
  ret i32 %0, !dbg !41
}

; Function Attrs: noinline uwtable
define internal void @_GLOBAL__sub_I_upstream_test.cpp() #0 section ".text.startup" !dbg !42 {
entry:
  call void @__cxx_global_var_init(), !dbg !44
  ret void
}

attributes #0 = { nounwind uwtable }
attributes #1 = { nounwind uwtable noinline optnone }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!17, !18, !19}
!llvm.ident = !{!20}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "glbl", scope: !2, file: !3, line: 7, type: !12, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "clang version 7.0.0 ", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5)
!3 = !DIFile(filename: "upstream_test.cpp", directory: "/home/mvoss/src/92562")
!4 = !{}
!5 = !{!0, !6, !13}
!6 = !DIGlobalVariableExpression(var: !7, expr: !DIExpression())
!7 = distinct !DIGlobalVariable(name: "n", scope: !2, file: !3, line: 8, type: !8, isLocal: false, isDefinition: true)!8 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "nested", scope: !9, file: !3, line: 6, size: 8, flags: DIFlagTypePassByValue, elements: !4, identifier: "_ZTSN6y_implIiE6nestedE")
!9 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "y_impl<int>", file: !3, line: 6, size: 8, flags: DIFlagTypePassByValue, elements: !4, templateParams: !10, identifier: "_ZTS6y_implIiE")
!10 = !{!11}
!11 = !DITemplateTypeParameter(type: !12)
!12 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!13 = !DIGlobalVariableExpression(var: !14, expr: !DIExpression())
!14 = distinct !DIGlobalVariable(name: "var", linkageName: "_Z3varIiE", scope: !2, file: !3, line: 1, type: !12, isLocal: false, isDefinition: true, templateParams: !15)
!15 = !{!16}
!16 = !DITemplateTypeParameter(name: "T", type: !12)
!17 = !{i32 2, !"Dwarf Version", i32 4}
!18 = !{i32 2, !"Debug Info Version", i32 3}
!19 = !{i32 1, !"wchar_size", i32 4}
!20 = !{!"clang version 7.0.0 "}
!21 = distinct !DISubprogram(name: "__cxx_global_var_init", scope: !3, file: !3, line: 7, type: !22, isLocal: true, isDefinition: true, scopeLine: 7, flags: DIFlagPrototyped, isOptimized: false, unit: !2, retainedNodes: !4)
!22 = !DISubroutineType(types: !23)
!23 = !{null}
!24 = !DILocation(line: 7, column: 12, scope: !21)
!25 = distinct !DISubprogram(name: "func<3, &glbl, y_impl, nullptr, 1, 2>", linkageName: "_Z4funcILi3EXadL_Z4glblEE6y_implLDn0EJLi1ELi2EEEiv", scope: !3, file: !3, line: 2, type: !26, isLocal: false, isDefinition: true, scopeLine: 2, flags: DIFlagPrototyped, isOptimized: false, unit: !2, templateParams: !28, retainedNodes: !4)
!26 = !DISubroutineType(types: !27)
!27 = !{!12}
!28 = !{!29, !30, !32, !33, !35}
!29 = !DITemplateValueParameter(name: "x", type: !12, value: i32 3)
!30 = !DITemplateValueParameter(type: !31, value: i32* @glbl)
!31 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !12, size: 64)
!32 = !DITemplateValueParameter(tag: DW_TAG_GNU_template_template_param, name: "y", value: !"y_impl")
!33 = !DITemplateValueParameter(name: "n", type: !34, value: i8 0)
!34 = !DIBasicType(tag: DW_TAG_unspecified_type, name: "decltype(nullptr)")
!35 = !DITemplateValueParameter(tag: DW_TAG_GNU_template_parameter_pack, name: "z", value: !36)
!36 = !{!37, !38}
!37 = !DITemplateValueParameter(type: !12, value: i32 1)
!38 = !DITemplateValueParameter(type: !12, value: i32 2)
!39 = !DILocation(line: 3, column: 12, scope: !25)
!40 = !DILocation(line: 4, column: 10, scope: !25)
!41 = !DILocation(line: 4, column: 3, scope: !25)
!42 = distinct !DISubprogram(linkageName: "_GLOBAL__sub_I_upstream_test.cpp", scope: !3, file: !3, type: !43, isLocal: true, isDefinition: true, flags: DIFlagArtificial, isOptimized: false, unit: !2, retainedNodes: !4)
!43 = !DISubroutineType(types: !4)
!44 = !DILocation(line: 0, scope: !42)
