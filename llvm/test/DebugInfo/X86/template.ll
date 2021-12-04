
; RUN: llc -mtriple=x86_64-linux -O0 -filetype=obj < %s | llvm-dwarfdump -v -debug-info - | FileCheck %s
; RUN: llc -mtriple=x86_64-linux -O0 -filetype=obj < %s | llvm-dwarfdump -verify - | FileCheck %s --check-prefix VERIFY

; IR generated with `clang++ -g -emit-llvm -S` from the following code:
; template<typename T> T var;
; enum e : unsigned char { E = (e)-1 };
; template<int x, int*, template<typename> class y, decltype(nullptr) n, e, int ...z>  int func() {
;  var<int> = 5;
;  return var<int>;
; }
; template<typename> struct y_impl { struct nested { }; };
; int glbl = func<3, &glbl, y_impl, nullptr, E, 1, 2>();
; y_impl<int>::nested n;

; VERIFY-NOT: error:

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


; CHECK: DW_AT_name{{.*}}"func<3, &glbl, y_impl, nullptr, E, 1, 2>"
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

; CHECK: DW_TAG_template_value_parameter
; CHECK: DW_AT_type [DW_FORM_ref4] {{.*}} "e")
; CHECK: DW_AT_const_value [DW_FORM_udata]     (255)

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

source_filename = "test.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%"struct.y_impl<int>::nested" = type { i8 }

$_Z4funcILi3EXadL_Z4glblEE6y_implLDn0EL1e255EJLi1ELi2EEEiv = comdat any

$_Z3varIiE = comdat any

@glbl = dso_local global i32 0, align 4, !dbg !0
@n = dso_local global %"struct.y_impl<int>::nested" zeroinitializer, align 1, !dbg !10
@_Z3varIiE = linkonce_odr dso_local global i32 0, comdat, align 4, !dbg !18
@llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @_GLOBAL__sub_I_test.cpp, i8* null }]

; Function Attrs: noinline uwtable
define internal void @__cxx_global_var_init() #0 section ".text.startup" !dbg !28 {
entry:
  %call = call i32 @_Z4funcILi3EXadL_Z4glblEE6y_implLDn0EL1e255EJLi1ELi2EEEiv(), !dbg !31
  store i32 %call, i32* @glbl, align 4, !dbg !32
  ret void, !dbg !31
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define linkonce_odr dso_local i32 @_Z4funcILi3EXadL_Z4glblEE6y_implLDn0EL1e255EJLi1ELi2EEEiv() #1 comdat !dbg !33 {
entry:
  store i32 5, i32* @_Z3varIiE, align 4, !dbg !48
  %0 = load i32, i32* @_Z3varIiE, align 4, !dbg !49
  ret i32 %0, !dbg !50
}

; Function Attrs: noinline uwtable
define internal void @_GLOBAL__sub_I_test.cpp() #0 section ".text.startup" !dbg !51 {
entry:
  call void @__cxx_global_var_init(), !dbg !53
  ret void
}

attributes #0 = { noinline uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { mustprogress noinline nounwind optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!22, !23, !24, !25, !26}
!llvm.ident = !{!27}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "glbl", scope: !2, file: !3, line: 8, type: !17, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !3, producer: "clang version 14.0.0 (git@github.com:llvm/llvm-project.git d0046d018934b284c995408a117371606472cd2a)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !9, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "test.cpp", directory: "")
!4 = !{!5}
!5 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "e", file: !3, line: 2, baseType: !6, size: 8, elements: !7, identifier: "_ZTS1e")
!6 = !DIBasicType(name: "unsigned char", size: 8, encoding: DW_ATE_unsigned_char)
!7 = !{!8}
!8 = !DIEnumerator(name: "E", value: 255, isUnsigned: true)
!9 = !{!0, !10, !18}
!10 = !DIGlobalVariableExpression(var: !11, expr: !DIExpression())
!11 = distinct !DIGlobalVariable(name: "n", scope: !2, file: !3, line: 9, type: !12, isLocal: false, isDefinition: true)
!12 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "nested", scope: !13, file: !3, line: 7, size: 8, flags: DIFlagTypePassByValue, elements: !14, identifier: "_ZTSN6y_implIiE6nestedE")
!13 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "y_impl<int>", file: !3, line: 7, size: 8, flags: DIFlagTypePassByValue, elements: !14, templateParams: !15, identifier: "_ZTS6y_implIiE")
!14 = !{}
!15 = !{!16}
!16 = !DITemplateTypeParameter(type: !17)
!17 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!18 = !DIGlobalVariableExpression(var: !19, expr: !DIExpression())
!19 = distinct !DIGlobalVariable(name: "var", linkageName: "_Z3varIiE", scope: !2, file: !3, line: 1, type: !17, isLocal: false, isDefinition: true, templateParams: !20)
!20 = !{!21}
!21 = !DITemplateTypeParameter(name: "T", type: !17)
!22 = !{i32 7, !"Dwarf Version", i32 4}
!23 = !{i32 2, !"Debug Info Version", i32 3}
!24 = !{i32 1, !"wchar_size", i32 4}
!25 = !{i32 7, !"uwtable", i32 1}
!26 = !{i32 7, !"frame-pointer", i32 2}
!27 = !{!"clang version 14.0.0 (git@github.com:llvm/llvm-project.git d0046d018934b284c995408a117371606472cd2a)"}
!28 = distinct !DISubprogram(name: "__cxx_global_var_init", scope: !3, file: !3, type: !29, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2, retainedNodes: !14)
!29 = !DISubroutineType(types: !30)
!30 = !{null}
!31 = !DILocation(line: 8, column: 12, scope: !28)
!32 = !DILocation(line: 0, scope: !28)
!33 = distinct !DISubprogram(name: "func<3, &glbl, y_impl, nullptr, E, 1, 2>", linkageName: "_Z4funcILi3EXadL_Z4glblEE6y_implLDn0EL1e255EJLi1ELi2EEEiv", scope: !3, file: !3, line: 3, type: !34, scopeLine: 3, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, templateParams: !36, retainedNodes: !14)
!34 = !DISubroutineType(types: !35)
!35 = !{!17}
!36 = !{!37, !38, !40, !41, !43, !44}
!37 = !DITemplateValueParameter(name: "x", type: !17, value: i32 3)
!38 = !DITemplateValueParameter(type: !39, value: i32* @glbl)
!39 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !17, size: 64)
!40 = !DITemplateValueParameter(tag: DW_TAG_GNU_template_template_param, name: "y", value: !"y_impl")
!41 = !DITemplateValueParameter(name: "n", type: !42, value: i8 0)
!42 = !DIBasicType(tag: DW_TAG_unspecified_type, name: "decltype(nullptr)")
!43 = !DITemplateValueParameter(type: !5, value: i8 -1)
!44 = !DITemplateValueParameter(tag: DW_TAG_GNU_template_parameter_pack, name: "z", value: !45)
!45 = !{!46, !47}
!46 = !DITemplateValueParameter(type: !17, value: i32 1)
!47 = !DITemplateValueParameter(type: !17, value: i32 2)
!48 = !DILocation(line: 4, column: 11, scope: !33)
!49 = !DILocation(line: 5, column: 9, scope: !33)
!50 = !DILocation(line: 5, column: 2, scope: !33)
!51 = distinct !DISubprogram(linkageName: "_GLOBAL__sub_I_test.cpp", scope: !3, file: !3, type: !52, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2, retainedNodes: !14)
!52 = !DISubroutineType(types: !14)
!53 = !DILocation(line: 0, scope: !51)
