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

define internal void @__cxx_global_var_init() section ".text.startup" {
entry:
  %call = call i32 @_Z4funcILi3EXadL_Z4glblEE6y_implLDn0EJLi1ELi2EEEiv(), !dbg !36
  store i32 %call, i32* @glbl, align 4, !dbg !36
  ret void, !dbg !36
}

; Function Attrs: nounwind uwtable
define linkonce_odr i32 @_Z4funcILi3EXadL_Z4glblEE6y_implLDn0EJLi1ELi2EEEiv() #0 {
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

!0 = !{!"0x11\004\00clang version 3.6.0 (trunk 224394) (llvm/trunk 224384)\000\00\000\00\001", !1, !2, !3, !9, !30, !2} ; [ DW_TAG_compile_unit ] [/tmp/dbginfo/template.cpp] [DW_LANG_C_plus_plus]
!1 = !{!"template.cpp", !"/tmp/dbginfo"}
!2 = !{}
!3 = !{!4, !8}
!4 = !{!"0x13\00y_impl<int>\002\008\008\000\000\000", !1, null, null, !2, null, !5, !"_ZTS6y_implIiE"} ; [ DW_TAG_structure_type ] [y_impl<int>] [line 2, size 8, align 8, offset 0] [def] [from ]
!5 = !{!6}
!6 = !{!"0x2f\00\000\000", null, !7, null}        ; [ DW_TAG_template_type_parameter ]
!7 = !{!"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!8 = !{!"0x13\00nested\002\008\008\000\000\000", !1, !"_ZTS6y_implIiE", null, !2, null, null, !"_ZTSN6y_implIiE6nestedE"} ; [ DW_TAG_structure_type ] [nested] [line 2, size 8, align 8, offset 0] [def] [from ]
!9 = !{!10, !14, !28}
!10 = !{!"0x2e\00__cxx_global_var_init\00__cxx_global_var_init\00\003\001\001\000\000\00256\000\003", !1, !11, !12, null, void ()* @__cxx_global_var_init, null, null, !2} ; [ DW_TAG_subprogram ] [line 3] [local] [def] [__cxx_global_var_init]
!11 = !{!"0x29", !1}                              ; [ DW_TAG_file_type ] [/tmp/dbginfo/template.cpp]
!12 = !{!"0x15\00\000\000\000\000\000\000", null, null, null, !13, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!13 = !{null}
!14 = !{!"0x2e\00func<3, &glbl, y_impl, nullptr, 1, 2>\00func<3, &glbl, y_impl, nullptr, 1, 2>\00_Z4funcILi3EXadL_Z4glblEE6y_implLDn0EJLi1ELi2EEEiv\001\000\001\000\000\00256\000\001", !1, !11, !15, null, i32 ()* @_Z4funcILi3EXadL_Z4glblEE6y_implLDn0EJLi1ELi2EEEiv, !17, null, !2} ; [ DW_TAG_subprogram ] [line 1] [def] [func<3, &glbl, y_impl, nullptr, 1, 2>]
!15 = !{!"0x15\00\000\000\000\000\000\000", null, null, null, !16, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!16 = !{!7}
!17 = !{!18, !19, !21, !22, !24}
!18 = !{!"0x30\00x\000\000", null, !7, i32 3, null} ; [ DW_TAG_template_value_parameter ]
!19 = !{!"0x30\00\000\000", null, !20, i32* @glbl, null} ; [ DW_TAG_template_value_parameter ]
!20 = !{!"0xf\00\000\0064\0064\000\000", null, null, !7} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from int]
!21 = !{!"0x4106\00y\000\000", null, null, !"y_impl", null} ; [ DW_TAG_GNU_template_template_param ]
!22 = !{!"0x30\00n\000\000", null, !23, i8 0, null} ; [ DW_TAG_template_value_parameter ]
!23 = !{!"0x3b\00decltype(nullptr)\000\000\000\000\000\000", null, null} ; [ DW_TAG_unspecified_type ] [decltype(nullptr)] [line 0, size 0, align 0, offset 0]
!24 = !{!"0x4107\00z\000\000", null, null, !25, null} ; [ DW_TAG_GNU_template_parameter_pack ]
!25 = !{!26, !27}
!26 = !{!"0x30\00\000\000", null, !7, i32 1, null} ; [ DW_TAG_template_value_parameter ]
!27 = !{!"0x30\00\000\000", null, !7, i32 2, null} ; [ DW_TAG_template_value_parameter ]
!28 = !{!"0x2e\00\00\00_GLOBAL__sub_I_template.cpp\000\001\001\000\000\0064\000\000", !1, !11, !29, null, void ()* @_GLOBAL__sub_I_template.cpp, null, null, !2} ; [ DW_TAG_subprogram ] [line 0] [local] [def]
!29 = !{!"0x15\00\000\000\000\000\000\000", null, null, null, !2, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!30 = !{!31, !32}
!31 = !{!"0x34\00glbl\00glbl\00\003\000\001", null, !11, !7, i32* @glbl, null} ; [ DW_TAG_variable ] [glbl] [line 3] [def]
!32 = !{!"0x34\00n\00n\00\004\000\001", null, !11, !"_ZTSN6y_implIiE6nestedE", %"struct.y_impl<int>::nested"* @n, null} ; [ DW_TAG_variable ] [n] [line 4] [def]
!33 = !{i32 2, !"Dwarf Version", i32 4}
!34 = !{i32 2, !"Debug Info Version", i32 2}
!35 = !{!"clang version 3.6.0 (trunk 224394) (llvm/trunk 224384)"}
!36 = !MDLocation(line: 3, column: 12, scope: !10)
!37 = !MDLocation(line: 1, column: 96, scope: !14)
!38 = !MDLocation(line: 0, scope: !28)
