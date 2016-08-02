; REQUIRES: object-emission

; RUN: llc -mtriple=x86_64-linux -O0 -filetype=obj < %s | llvm-dwarfdump -debug-dump=info - | FileCheck %s

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

; CHECK-NEXT: DW_AT_location [DW_FORM_exprloc]{{ *}}(<0xa> 03 00 00 00 00 00 00 00 00 9f )
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

%"struct.y_impl<int>::nested" = type { i8 }

@glbl = global i32 0, align 4
@n = global %"struct.y_impl<int>::nested" zeroinitializer, align 1
@llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @_GLOBAL__sub_I_template.cpp, i8* null }]

define internal void @__cxx_global_var_init() section ".text.startup" !dbg !10 {
entry:
  %call = call i32 @_Z4funcILi3EXadL_Z4glblEE6y_implLDn0EJLi1ELi2EEEiv(), !dbg !36
  store i32 %call, i32* @glbl, align 4, !dbg !36
  ret void, !dbg !36
}

; Function Attrs: nounwind uwtable
define linkonce_odr i32 @_Z4funcILi3EXadL_Z4glblEE6y_implLDn0EJLi1ELi2EEEiv() #0 !dbg !14 {
entry:
  ret i32 3, !dbg !37
}

define internal void @_GLOBAL__sub_I_template.cpp() section ".text.startup" {
entry:
  call void @__cxx_global_var_init(), !dbg !38
  ret void
}

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!33, !34}
!llvm.ident = !{!35}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.6.0 (trunk 224394) (llvm/trunk 224384)", isOptimized: false, emissionKind: 1, file: !1, enums: !2, retainedTypes: !3, subprograms: !9, globals: !30, imports: !2)
!1 = !DIFile(filename: "template.cpp", directory: "/tmp/dbginfo")
!2 = !{}
!3 = !{!4, !8}
!4 = !DICompositeType(tag: DW_TAG_structure_type, name: "y_impl<int>", line: 2, size: 8, align: 8, file: !1, elements: !2, templateParams: !5, identifier: "_ZTS6y_implIiE")
!5 = !{!6}
!6 = !DITemplateTypeParameter(type: !7)
!7 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!8 = !DICompositeType(tag: DW_TAG_structure_type, name: "nested", line: 2, size: 8, align: 8, file: !1, scope: !"_ZTS6y_implIiE", elements: !2, identifier: "_ZTSN6y_implIiE6nestedE")
!9 = !{!10, !14, !28}
!10 = distinct !DISubprogram(name: "__cxx_global_var_init", line: 3, isLocal: true, isDefinition: true, flags: DIFlagPrototyped, isOptimized: false, scopeLine: 3, file: !1, scope: !11, type: !12, variables: !2)
!11 = !DIFile(filename: "template.cpp", directory: "/tmp/dbginfo")
!12 = !DISubroutineType(types: !13)
!13 = !{null}
!14 = distinct !DISubprogram(name: "func<3, &glbl, y_impl, nullptr, 1, 2>", linkageName: "_Z4funcILi3EXadL_Z4glblEE6y_implLDn0EJLi1ELi2EEEiv", line: 1, isLocal: false, isDefinition: true, flags: DIFlagPrototyped, isOptimized: false, scopeLine: 1, file: !1, scope: !11, type: !15, templateParams: !17, variables: !2)
!15 = !DISubroutineType(types: !16)
!16 = !{!7}
!17 = !{!18, !19, !21, !22, !24}
!18 = !DITemplateValueParameter(tag: DW_TAG_template_value_parameter, name: "x", type: !7, value: i32 3)
!19 = !DITemplateValueParameter(tag: DW_TAG_template_value_parameter, type: !20, value: i32* @glbl)
!20 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, baseType: !7)
!21 = !DITemplateValueParameter(tag: DW_TAG_GNU_template_template_param, name: "y", type: null, value: !"y_impl")
!22 = !DITemplateValueParameter(tag: DW_TAG_template_value_parameter, name: "n", type: !23, value: i8 0)
!23 = !DIBasicType(tag: DW_TAG_unspecified_type, name: "decltype(nullptr)")
!24 = !DITemplateValueParameter(tag: DW_TAG_GNU_template_parameter_pack, name: "z", type: null, value: !25)
!25 = !{!26, !27}
!26 = !DITemplateValueParameter(tag: DW_TAG_template_value_parameter, type: !7, value: i32 1)
!27 = !DITemplateValueParameter(tag: DW_TAG_template_value_parameter, type: !7, value: i32 2)
!28 = distinct !DISubprogram(name: "", linkageName: "_GLOBAL__sub_I_template.cpp", isLocal: true, isDefinition: true, flags: DIFlagArtificial, isOptimized: false, file: !1, scope: !11, type: !29, variables: !2)
!29 = !DISubroutineType(types: !2)
!30 = !{!31, !32}
!31 = !DIGlobalVariable(name: "glbl", line: 3, isLocal: false, isDefinition: true, scope: null, file: !11, type: !7, variable: i32* @glbl)
!32 = !DIGlobalVariable(name: "n", line: 4, isLocal: false, isDefinition: true, scope: null, file: !11, type: !"_ZTSN6y_implIiE6nestedE", variable: %"struct.y_impl<int>::nested"* @n)
!33 = !{i32 2, !"Dwarf Version", i32 4}
!34 = !{i32 2, !"Debug Info Version", i32 3}
!35 = !{!"clang version 3.6.0 (trunk 224394) (llvm/trunk 224384)"}
!36 = !DILocation(line: 3, column: 12, scope: !10)
!37 = !DILocation(line: 1, column: 96, scope: !14)
!38 = !DILocation(line: 0, scope: !28)
