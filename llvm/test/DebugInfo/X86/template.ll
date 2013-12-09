; REQUIRES: object-emission

; RUN: llc -mtriple=x86_64-linux -O0 -filetype=obj < %s > %t
; RUN: llvm-dwarfdump %t | FileCheck %s

; IR generated with `clang++ -g -emit-llvm -S` from the following code:
; template<int x, int*, template<typename> class y, int ...z>  int func() { return 3; }
; template<typename> struct y_impl { struct nested { }; };
; int glbl = func<3, &glbl, y_impl, 1, 2>();
; y_impl<int>::nested n;

; CHECK: [[INT:0x[0-9a-f]*]]:{{ *}}DW_TAG_base_type
; CHECK-NEXT: DW_AT_name{{.*}} = "int"

; CHECK: DW_TAG_structure_type
; CHECK-NEXT: DW_AT_name{{.*}}"y_impl<int>"
; CHECK-NOT: NULL
; CHECK: DW_TAG_template_type_parameter

; CHECK: DW_AT_name{{.*}}"func<3, &glbl, y_impl, 1, 2>"
; CHECK-NOT: NULL
; CHECK: DW_TAG_template_value_parameter
; CHECK-NEXT: DW_AT_type{{.*}}=> {[[INT]]}
; CHECK-NEXT: DW_AT_name{{.*}}= "x"

; This could be made shorter by encoding it as _sdata rather than data4, or
; even as data1. DWARF strongly urges implementations to prefer 
; _sdata/_udata rather than dataN

; CHECK-NEXT: DW_AT_const_value [DW_FORM_sdata]{{.*}}(3)

; CHECK: DW_TAG_template_value_parameter
; CHECK-NEXT: DW_AT_type{{.*}}=> {[[INTPTR:0x[0-9a-f]*]]}

; The address of the global 'glbl', followed by DW_OP_stack_value (9f), to use
; the value immediately, rather than indirecting through the address.

; CHECK-NEXT: DW_AT_location [DW_FORM_block1]{{ *}}(<0x0a> 03 00 00 00 00 00 00 00 00 9f )
; CHECK-NOT: NULL

; CHECK: DW_TAG_GNU_template_template_param
; CHECK-NEXT: DW_AT_name{{.*}}= "y"
; CHECK-NEXT: DW_AT_GNU_template_name{{.*}}= "y_impl"
; CHECK-NOT: NULL

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

%"struct.y_impl<int>::nested" = type { i8 }

@glbl = global i32 0, align 4
@n = global %"struct.y_impl<int>::nested" zeroinitializer, align 1
@llvm.global_ctors = appending global [1 x { i32, void ()* }] [{ i32, void ()* } { i32 65535, void ()* @_GLOBAL__I_a }]

define internal void @__cxx_global_var_init() section ".text.startup" {
entry:
  %call = call i32 @_Z4funcILi3EXadL_Z4glblEE6y_implJLi1ELi2EEEiv(), !dbg !33
  store i32 %call, i32* @glbl, align 4, !dbg !33
  ret void, !dbg !33
}

; Function Attrs: nounwind uwtable
define linkonce_odr i32 @_Z4funcILi3EXadL_Z4glblEE6y_implJLi1ELi2EEEiv() #0 {
entry:
  ret i32 3, !dbg !34
}

define internal void @_GLOBAL__I_a() section ".text.startup" {
entry:
  call void @__cxx_global_var_init(), !dbg !35
  ret void, !dbg !35
}

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!31, !36}
!llvm.ident = !{!32}

!0 = metadata !{i32 786449, metadata !1, i32 4, metadata !"clang version 3.4 (trunk 192849) (llvm/trunk 192850)", i1 false, metadata !"", i32 0, metadata !2, metadata !3, metadata !9, metadata !28, metadata !2, metadata !""} ; [ DW_TAG_compile_unit ] [/usr/local/google/home/echristo/tmp/bar.cpp] [DW_LANG_C_plus_plus]
!1 = metadata !{metadata !"bar.cpp", metadata !"/usr/local/google/home/echristo/tmp"}
!2 = metadata !{i32 0}
!3 = metadata !{metadata !4, metadata !8}
!4 = metadata !{i32 786451, metadata !1, null, metadata !"y_impl<int>", i32 2, i64 8, i64 8, i32 0, i32 0, null, metadata !2, i32 0, null, metadata !5, metadata !"_ZTS6y_implIiE"} ; [ DW_TAG_structure_type ] [y_impl<int>] [line 2, size 8, align 8, offset 0] [def] [from ]
!5 = metadata !{metadata !6}
!6 = metadata !{i32 786479, null, metadata !"", metadata !7, null, i32 0, i32 0} ; [ DW_TAG_template_type_parameter ]
!7 = metadata !{i32 786468, null, null, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!8 = metadata !{i32 786451, metadata !1, metadata !"_ZTS6y_implIiE", metadata !"nested", i32 2, i64 8, i64 8, i32 0, i32 0, null, metadata !2, i32 0, null, null, metadata !"_ZTSN6y_implIiE6nestedE"} ; [ DW_TAG_structure_type ] [nested] [line 2, size 8, align 8, offset 0] [def] [from ]
!9 = metadata !{metadata !10, metadata !14, metadata !26}
!10 = metadata !{i32 786478, metadata !1, metadata !11, metadata !"__cxx_global_var_init", metadata !"__cxx_global_var_init", metadata !"", i32 3, metadata !12, i1 true, i1 true, i32 0, i32 0, null, i32 256, i1 false, void ()* @__cxx_global_var_init, null, null, metadata !2, i32 3} ; [ DW_TAG_subprogram ] [line 3] [local] [def] [__cxx_global_var_init]
!11 = metadata !{i32 786473, metadata !1}         ; [ DW_TAG_file_type ] [/usr/local/google/home/echristo/tmp/bar.cpp]
!12 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !13, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!13 = metadata !{null}
!14 = metadata !{i32 786478, metadata !1, metadata !11, metadata !"func<3, &glbl, y_impl, 1, 2>", metadata !"func<3, &glbl, y_impl, 1, 2>", metadata !"_Z4funcILi3EXadL_Z4glblEE6y_implJLi1ELi2EEEiv", i32 1, metadata !15, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, i32 ()* @_Z4funcILi3EXadL_Z4glblEE6y_implJLi1ELi2EEEiv, metadata !17, null, metadata !2, i32 1} ; [ DW_TAG_subprogram ] [line 1] [def] [func<3, &glbl, y_impl, 1, 2>]
!15 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !16, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!16 = metadata !{metadata !7}
!17 = metadata !{metadata !18, metadata !19, metadata !21, metadata !22}
!18 = metadata !{i32 786480, null, metadata !"x", metadata !7, i32 3, null, i32 0, i32 0} ; [ DW_TAG_template_value_parameter ]
!19 = metadata !{i32 786480, null, metadata !"", metadata !20, i32* @glbl, null, i32 0, i32 0} ; [ DW_TAG_template_value_parameter ]
!20 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 0, metadata !7} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from int]
!21 = metadata !{i32 803078, null, metadata !"y", null, metadata !"y_impl", null, i32 0, i32 0} ; [ DW_TAG_GNU_template_template_param ]
!22 = metadata !{i32 803079, null, metadata !"z", null, metadata !23, null, i32 0, i32 0} ; [ DW_TAG_GNU_template_parameter_pack ]
!23 = metadata !{metadata !24, metadata !25}
!24 = metadata !{i32 786480, null, metadata !"", metadata !7, i32 1, null, i32 0, i32 0} ; [ DW_TAG_template_value_parameter ]
!25 = metadata !{i32 786480, null, metadata !"", metadata !7, i32 2, null, i32 0, i32 0} ; [ DW_TAG_template_value_parameter ]
!26 = metadata !{i32 786478, metadata !1, metadata !11, metadata !"", metadata !"", metadata !"_GLOBAL__I_a", i32 1, metadata !27, i1 true, i1 true, i32 0, i32 0, null, i32 64, i1 false, void ()* @_GLOBAL__I_a, null, null, metadata !2, i32 1} ; [ DW_TAG_subprogram ] [line 1] [local] [def]
!27 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !2, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!28 = metadata !{metadata !29, metadata !30}
!29 = metadata !{i32 786484, i32 0, null, metadata !"glbl", metadata !"glbl", metadata !"", metadata !11, i32 3, metadata !7, i32 0, i32 1, i32* @glbl, null} ; [ DW_TAG_variable ] [glbl] [line 3] [def]
!30 = metadata !{i32 786484, i32 0, null, metadata !"n", metadata !"n", metadata !"", metadata !11, i32 4, metadata !8, i32 0, i32 1, %"struct.y_impl<int>::nested"* @n, null} ; [ DW_TAG_variable ] [n] [line 4] [def]
!31 = metadata !{i32 2, metadata !"Dwarf Version", i32 4}
!32 = metadata !{metadata !"clang version 3.4 (trunk 192849) (llvm/trunk 192850)"}
!33 = metadata !{i32 3, i32 0, metadata !10, null}
!34 = metadata !{i32 1, i32 0, metadata !14, null}
!35 = metadata !{i32 1, i32 0, metadata !26, null}
!36 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
