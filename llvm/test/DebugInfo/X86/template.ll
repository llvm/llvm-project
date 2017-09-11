; REQUIRES: object-emission

; RUN: llc -mtriple=x86_64-linux -O0 -filetype=obj < %s | llvm-dwarfdump -v -debug-info - | FileCheck %s

; IR generated with `clang++ -g -emit-llvm -S` from the following code:
; template<int x, int*, template<typename> class y, decltype(nullptr) n, int ...z>  int func() { return 3; }
; template<typename> struct y_impl { struct nested { }; };
; int glbl = func<3, &glbl, y_impl, nullptr, 1, 2>();
; y_impl<int>::nested n;

; CHECK: [[INT:0x[0-9a-f]*]]:{{ *}}DW_TAG_base_type
; CHECK-NEXT: DW_AT_name{{.*}} = "int"

; CHECK: DW_TAG_structure_type
; CHECK-NEXT: DW_AT_name{{.*}}"y_impl<int>"
; CHECK-NOT: NULL
; CHECK: DW_TAG_template_type_parameter

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

@glbl = global i32 0, align 4, !dbg !0
@n = global %"struct.y_impl<int>::nested" zeroinitializer, align 1, !dbg !4
@llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @_GLOBAL__sub_I_template.cpp, i8* null }]

define internal void @__cxx_global_var_init() section ".text.startup" !dbg !17 {
entry:
  %call = call i32 @_Z4funcILi3EXadL_Z4glblEE6y_implLDn0EJLi1ELi2EEEiv(), !dbg !20
  store i32 %call, i32* @glbl, align 4, !dbg !20
  ret void, !dbg !20
}

; Function Attrs: nounwind uwtable
define linkonce_odr i32 @_Z4funcILi3EXadL_Z4glblEE6y_implLDn0EJLi1ELi2EEEiv() #0 !dbg !21 {
entry:
  ret i32 3, !dbg !35
}

define internal void @_GLOBAL__sub_I_template.cpp() section ".text.startup" {
entry:
  call void @__cxx_global_var_init(), !dbg !36
  ret void
}

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!11}
!llvm.module.flags = !{!14, !15}
!llvm.ident = !{!16}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = !DIGlobalVariable(name: "glbl", scope: null, file: !2, line: 3, type: !3, isLocal: false, isDefinition: true)
!2 = !DIFile(filename: "template.cpp", directory: "/tmp/dbginfo")
!3 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!4 = !DIGlobalVariableExpression(var: !5, expr: !DIExpression())
!5 = !DIGlobalVariable(name: "n", scope: null, file: !2, line: 4, type: !6, isLocal: false, isDefinition: true)
!6 = !DICompositeType(tag: DW_TAG_structure_type, name: "nested", scope: !7, file: !2, line: 2, size: 8, align: 8, elements: !8, identifier: "_ZTSN6y_implIiE6nestedE")
!7 = !DICompositeType(tag: DW_TAG_structure_type, name: "y_impl<int>", file: !2, line: 2, size: 8, align: 8, elements: !8, templateParams: !9, identifier: "_ZTS6y_implIiE")
!8 = !{}
!9 = !{!10}
!10 = !DITemplateTypeParameter(type: !3)
!11 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !2, producer: "clang version 3.6.0 (trunk 224394) (llvm/trunk 224384)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !8, retainedTypes: !12, globals: !13, imports: !8)
!12 = !{!7, !6}
!13 = !{!0, !4}
!14 = !{i32 2, !"Dwarf Version", i32 4}
!15 = !{i32 2, !"Debug Info Version", i32 3}
!16 = !{!"clang version 3.6.0 (trunk 224394) (llvm/trunk 224384)"}
!17 = distinct !DISubprogram(name: "__cxx_global_var_init", scope: !2, file: !2, line: 3, type: !18, isLocal: true, isDefinition: true, scopeLine: 3, flags: DIFlagPrototyped, isOptimized: false, unit: !11, variables: !8)
!18 = !DISubroutineType(types: !19)
!19 = !{null}
!20 = !DILocation(line: 3, column: 12, scope: !17)
!21 = distinct !DISubprogram(name: "func<3, &glbl, y_impl, nullptr, 1, 2>", linkageName: "_Z4funcILi3EXadL_Z4glblEE6y_implLDn0EJLi1ELi2EEEiv", scope: !2, file: !2, line: 1, type: !22, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: false, unit: !11, templateParams: !24, variables: !8)
!22 = !DISubroutineType(types: !23)
!23 = !{!3}
!24 = !{!25, !26, !28, !29, !31}
!25 = !DITemplateValueParameter(name: "x", type: !3, value: i32 3)
!26 = !DITemplateValueParameter(type: !27, value: i32* @glbl)
!27 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !3, size: 64, align: 64)
!28 = !DITemplateValueParameter(tag: DW_TAG_GNU_template_template_param, name: "y", value: !"y_impl")
!29 = !DITemplateValueParameter(name: "n", type: !30, value: i8 0)
!30 = !DIBasicType(tag: DW_TAG_unspecified_type, name: "decltype(nullptr)")
!31 = !DITemplateValueParameter(tag: DW_TAG_GNU_template_parameter_pack, name: "z", value: !32)
!32 = !{!33, !34}
!33 = !DITemplateValueParameter(type: !3, value: i32 1)
!34 = !DITemplateValueParameter(type: !3, value: i32 2)
!35 = !DILocation(line: 1, column: 96, scope: !21)
!36 = !DILocation(line: 0, scope: !37)
!37 = distinct !DISubprogram(linkageName: "_GLOBAL__sub_I_template.cpp", scope: !2, file: !2, type: !38, isLocal: true, isDefinition: true, flags: DIFlagArtificial, isOptimized: false, unit: !11, variables: !8)
!38 = !DISubroutineType(types: !8)

