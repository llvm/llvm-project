; RUN: llc -mtriple=x86_64-pc-linux-gnu -generate-gnu-dwarf-pub-sections < %s | FileCheck -check-prefix=ASM %s
; RUN: llc -mtriple=x86_64-pc-linux-gnu -generate-gnu-dwarf-pub-sections -filetype=obj < %s | llvm-dwarfdump - | FileCheck %s
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
;   struct D {
;     int A;
;   } d;
; }

; ASM: .section        .debug_gnu_pubnames
; ASM: .byte   32                      # Kind: VARIABLE, EXTERNAL
; ASM-NEXT: .asciz  "global_variable"       # External Name

; ASM: .section        .debug_gnu_pubtypes
; ASM: .byte   16                      # Kind: TYPE, EXTERNAL
; ASM-NEXT: .asciz  "C"                     # External Name

; CHECK: .debug_info contents:
; CHECK: DW_AT_GNU_pubnames [DW_FORM_sec_offset]   (0x00000000)
; CHECK: DW_AT_GNU_pubtypes [DW_FORM_sec_offset]   (0x00000000)

; CHECK: [[C:[0-9a-f]+]]: DW_TAG_structure_type
; CHECK-NEXT: DW_AT_name {{.*}} "C"

; CHECK: [[STATIC_MEM_DECL:[0-9a-f]+]]: DW_TAG_member
; CHECK-NEXT: DW_AT_name {{.*}} "static_member_variable"

; CHECK: [[MEM_FUNC_DECL:[0-9a-f]+]]: DW_TAG_subprogram
; CHECK-NEXT: DW_AT_MIPS_linkage_name
; CHECK-NEXT: DW_AT_name {{.*}} "member_function"

; CHECK: [[STATIC_MEM_FUNC_DECL:[0-9a-f]+]]: DW_TAG_subprogram
; CHECK-NEXT: DW_AT_MIPS_linkage_name
; CHECK-NEXT: DW_AT_name {{.*}} "static_member_function"

; CHECK: [[INT:[0-9a-f]+]]: DW_TAG_base_type
; CHECK-NEXT: DW_AT_name {{.*}} "int"

; CHECK: [[STATIC_MEM_VAR:[0-9a-f]+]]: DW_TAG_variable
; CHECK-NEXT: DW_AT_specification {{.*}}[[STATIC_MEM_DECL]]

; CHECK: [[GLOB_VAR:[0-9a-f]+]]: DW_TAG_variable
; CHECK-NEXT: DW_AT_name {{.*}} "global_variable"

; CHECK: [[NS:[0-9a-f]+]]: DW_TAG_namespace
; CHECK-NEXT: DW_AT_name {{.*}} "ns"

; CHECK: [[GLOB_NS_VAR_DECL:[0-9a-f]+]]: DW_TAG_variable
; CHECK-NEXT: DW_AT_name {{.*}} "global_namespace_variable"

; CHECK: [[D_VAR_DECL:[0-9a-f]+]]: DW_TAG_variable
; CHECK-NEXT: DW_AT_name {{.*}} "d"

; CHECK: [[D:[0-9a-f]+]]: DW_TAG_structure_type
; CHECK-NEXT: DW_AT_name {{.*}} "D"

; CHECK: [[GLOB_NS_FUNC:[0-9a-f]+]]: DW_TAG_subprogram
; CHECK-NEXT: DW_AT_MIPS_linkage_name
; CHECK-NEXT: DW_AT_name {{.*}} "global_namespace_function"

; CHECK: [[GLOB_NS_VAR:[0-9a-f]+]]: DW_TAG_variable
; CHECK-NEXT: DW_AT_specification {{.*}}[[GLOB_NS_VAR_DECL]]

; CHECK: [[D_VAR:[0-9a-f]+]]: DW_TAG_variable
; CHECK-NEXT: DW_AT_specification {{.*}}[[D_VAR_DECL]]

; CHECK: [[MEM_FUNC:[0-9a-f]+]]: DW_TAG_subprogram
; CHECK-NEXT: DW_AT_specification {{.*}}[[MEM_FUNC_DECL]]

; CHECK: [[STATIC_MEM_FUNC:[0-9a-f]+]]: DW_TAG_subprogram
; CHECK-NEXT: DW_AT_specification {{.*}}[[STATIC_MEM_FUNC_DECL]]

; CHECK: [[GLOBAL_FUNC:[0-9a-f]+]]: DW_TAG_subprogram
; CHECK-NEXT: DW_AT_MIPS_linkage_name
; CHECK-NEXT: DW_AT_name {{.*}} "global_function"

; CHECK-LABEL: .debug_gnu_pubnames contents:
; CHECK-NEXT: length = 0x000000e7 version = 0x0002 unit_offset = 0x00000000 unit_size = 0x0000017b
; CHECK-NEXT: Offset     Linkage  Kind     Name
; CHECK-DAG:  [[GLOBAL_FUNC]] EXTERNAL FUNCTION "global_function"
; CHECK-DAG:  [[NS]] EXTERNAL TYPE     "ns"
; CHECK-DAG:  [[MEM_FUNC]] EXTERNAL FUNCTION "C::member_function"
; CHECK-DAG:  [[GLOB_VAR]] EXTERNAL VARIABLE "global_variable"
; CHECK-DAG:  [[GLOB_NS_VAR]] EXTERNAL VARIABLE "ns::global_namespace_variable"
; CHECK-DAG:  [[GLOB_NS_FUNC]] EXTERNAL FUNCTION "ns::global_namespace_function"
; CHECK-DAG:  [[D_VAR]] EXTERNAL VARIABLE "ns::d"
; CHECK-DAG:  [[STATIC_MEM_VAR]] EXTERNAL VARIABLE "C::static_member_variable"
; CHECK-DAG:  [[STATIC_MEM_FUNC]] EXTERNAL FUNCTION "C::static_member_function"


; CHECK-LABEL: debug_gnu_pubtypes contents:
; CHECK: Offset     Linkage  Kind     Name
; CHECK-DAG:  [[C]] EXTERNAL TYPE     "C"
; CHECK-DAG:  [[D]] EXTERNAL TYPE     "ns::D"
; CHECK-DAG:  [[INT]] STATIC   TYPE     "int"

%struct.C = type { i8 }
%"struct.ns::D" = type { i32 }

@_ZN1C22static_member_variableE = global i32 0, align 4
@global_variable = global %struct.C zeroinitializer, align 1
@_ZN2ns25global_namespace_variableE = global i32 1, align 4
@_ZN2ns1dE = global %"struct.ns::D" zeroinitializer, align 4

; Function Attrs: nounwind uwtable
define void @_ZN1C15member_functionEv(%struct.C* %this) #0 align 2 {
entry:
  %this.addr = alloca %struct.C*, align 8
  store %struct.C* %this, %struct.C** %this.addr, align 8
  call void @llvm.dbg.declare(metadata !{%struct.C** %this.addr}, metadata !36), !dbg !38
  %this1 = load %struct.C** %this.addr
  store i32 0, i32* @_ZN1C22static_member_variableE, align 4, !dbg !39
  ret void, !dbg !39
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata) #1

; Function Attrs: nounwind uwtable
define i32 @_ZN1C22static_member_functionEv() #0 align 2 {
entry:
  %0 = load i32* @_ZN1C22static_member_variableE, align 4, !dbg !40
  ret i32 %0, !dbg !40
}

; Function Attrs: nounwind uwtable
define i32 @_Z15global_functionv() #0 {
entry:
  ret i32 -1, !dbg !41
}

; Function Attrs: nounwind uwtable
define void @_ZN2ns25global_namespace_functionEv() #0 {
entry:
  call void @_ZN1C15member_functionEv(%struct.C* @global_variable), !dbg !42
  ret void, !dbg !42
}

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!34, !43}
!llvm.ident = !{!35}

!0 = metadata !{i32 786449, metadata !1, i32 4, metadata !"clang version 3.4 (trunk 192862) (llvm/trunk 192861)", i1 false, metadata !"", i32 0, metadata !2, metadata !3, metadata !21, metadata !29, metadata !2, metadata !""} ; [ DW_TAG_compile_unit ] [/usr/local/google/home/echristo/tmp/pubnames.cpp] [DW_LANG_C_plus_plus]
!1 = metadata !{metadata !"pubnames.cpp", metadata !"/usr/local/google/home/echristo/tmp"}
!2 = metadata !{i32 0}
!3 = metadata !{metadata !4, metadata !17}
!4 = metadata !{i32 786451, metadata !1, null, metadata !"C", i32 1, i64 8, i64 8, i32 0, i32 0, null, metadata !5, i32 0, null, null, metadata !"_ZTS1C"} ; [ DW_TAG_structure_type ] [C] [line 1, size 8, align 8, offset 0] [def] [from ]
!5 = metadata !{metadata !6, metadata !8, metadata !13}
!6 = metadata !{i32 786445, metadata !1, metadata !"_ZTS1C", metadata !"static_member_variable", i32 4, i64 0, i64 0, i64 0, i32 4096, metadata !7, null} ; [ DW_TAG_member ] [static_member_variable] [line 4, size 0, align 0, offset 0] [static] [from int]
!7 = metadata !{i32 786468, null, null, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!8 = metadata !{i32 786478, metadata !1, metadata !"_ZTS1C", metadata !"member_function", metadata !"member_function", metadata !"_ZN1C15member_functionEv", i32 2, metadata !9, i1 false, i1 false, i32 0, i32 0, null, i32 256, i1 false, null, null, i32 0, metadata !12, i32 2} ; [ DW_TAG_subprogram ] [line 2] [member_function]
!9 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !10, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!10 = metadata !{null, metadata !11}
!11 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 1088, metadata !"_ZTS1C"} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from _ZTS1C]
!12 = metadata !{i32 786468}
!13 = metadata !{i32 786478, metadata !1, metadata !"_ZTS1C", metadata !"static_member_function", metadata !"static_member_function", metadata !"_ZN1C22static_member_functionEv", i32 3, metadata !14, i1 false, i1 false, i32 0, i32 0, null, i32 256, i1 false, null, null, i32 0, metadata !16, i32 3} ; [ DW_TAG_subprogram ] [line 3] [static_member_function]
!14 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !15, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!15 = metadata !{metadata !7}
!16 = metadata !{i32 786468}
!17 = metadata !{i32 786451, metadata !1, metadata !18, metadata !"D", i32 21, i64 32, i64 32, i32 0, i32 0, null, metadata !19, i32 0, null, null, metadata !"_ZTSN2ns1DE"} ; [ DW_TAG_structure_type ] [D] [line 21, size 32, align 32, offset 0] [def] [from ]
!18 = metadata !{i32 786489, metadata !1, null, metadata !"ns", i32 17} ; [ DW_TAG_namespace ] [ns] [line 17]
!19 = metadata !{metadata !20}
!20 = metadata !{i32 786445, metadata !1, metadata !"_ZTSN2ns1DE", metadata !"A", i32 22, i64 32, i64 32, i64 0, i32 0, metadata !7} ; [ DW_TAG_member ] [A] [line 22, size 32, align 32, offset 0] [from int]
!21 = metadata !{metadata !22, metadata !23, metadata !24, metadata !26}
!22 = metadata !{i32 786478, metadata !1, metadata !"_ZTS1C", metadata !"member_function", metadata !"member_function", metadata !"_ZN1C15member_functionEv", i32 9, metadata !9, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, void (%struct.C*)* @_ZN1C15member_functionEv, null, metadata !8, metadata !2, i32 9} ; [ DW_TAG_subprogram ] [line 9] [def] [member_function]
!23 = metadata !{i32 786478, metadata !1, metadata !"_ZTS1C", metadata !"static_member_function", metadata !"static_member_function", metadata !"_ZN1C22static_member_functionEv", i32 11, metadata !14, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, i32 ()* @_ZN1C22static_member_functionEv, null, metadata !13, metadata !2, i32 11} ; [ DW_TAG_subprogram ] [line 11] [def] [static_member_function]
!24 = metadata !{i32 786478, metadata !1, metadata !25, metadata !"global_function", metadata !"global_function", metadata !"_Z15global_functionv", i32 15, metadata !14, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, i32 ()* @_Z15global_functionv, null, null, metadata !2, i32 15} ; [ DW_TAG_subprogram ] [line 15] [def] [global_function]
!25 = metadata !{i32 786473, metadata !1}         ; [ DW_TAG_file_type ] [/usr/local/google/home/echristo/tmp/pubnames.cpp]
!26 = metadata !{i32 786478, metadata !1, metadata !18, metadata !"global_namespace_function", metadata !"global_namespace_function", metadata !"_ZN2ns25global_namespace_functionEv", i32 18, metadata !27, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, void ()* @_ZN2ns25global_namespace_functionEv, null, null, metadata !2, i32 18} ; [ DW_TAG_subprogram ] [line 18] [def] [global_namespace_function]
!27 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !28, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!28 = metadata !{null}
!29 = metadata !{metadata !30, metadata !31, metadata !32, metadata !33}
!30 = metadata !{i32 786484, i32 0, metadata !4, metadata !"static_member_variable", metadata !"static_member_variable", metadata !"_ZN1C22static_member_variableE", metadata !25, i32 7, metadata !7, i32 0, i32 1, i32* @_ZN1C22static_member_variableE, metadata !6} ; [ DW_TAG_variable ] [static_member_variable] [line 7] [def]
!31 = metadata !{i32 786484, i32 0, null, metadata !"global_variable", metadata !"global_variable", metadata !"", metadata !25, i32 13, metadata !4, i32 0, i32 1, %struct.C* @global_variable, null} ; [ DW_TAG_variable ] [global_variable] [line 13] [def]
!32 = metadata !{i32 786484, i32 0, metadata !18, metadata !"global_namespace_variable", metadata !"global_namespace_variable", metadata !"_ZN2ns25global_namespace_variableE", metadata !25, i32 19, metadata !7, i32 0, i32 1, i32* @_ZN2ns25global_namespace_variableE, null} ; [ DW_TAG_variable ] [global_namespace_variable] [line 19] [def]
!33 = metadata !{i32 786484, i32 0, metadata !18, metadata !"d", metadata !"d", metadata !"_ZN2ns1dE", metadata !25, i32 23, metadata !17, i32 0, i32 1, %"struct.ns::D"* @_ZN2ns1dE, null} ; [ DW_TAG_variable ] [d] [line 23] [def]
!34 = metadata !{i32 2, metadata !"Dwarf Version", i32 4}
!35 = metadata !{metadata !"clang version 3.4 (trunk 192862) (llvm/trunk 192861)"}
!36 = metadata !{i32 786689, metadata !22, metadata !"this", null, i32 16777216, metadata !37, i32 1088, i32 0} ; [ DW_TAG_arg_variable ] [this] [line 0]
!37 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 0, metadata !"_ZTS1C"} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from _ZTS1C]
!38 = metadata !{i32 0, i32 0, metadata !22, null}
!39 = metadata !{i32 9, i32 0, metadata !22, null}
!40 = metadata !{i32 11, i32 0, metadata !23, null}
!41 = metadata !{i32 15, i32 0, metadata !24, null}
!42 = metadata !{i32 18, i32 0, metadata !26, null}

!43 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
