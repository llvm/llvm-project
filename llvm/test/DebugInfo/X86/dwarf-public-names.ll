; RUN: llc -mtriple=x86_64-pc-linux-gnu -filetype=obj -o %t.o < %s
; RUN: llvm-dwarfdump -debug-dump=pubnames %t.o | FileCheck --check-prefix=LINUX %s
; RUN: llc -mtriple=x86_64-apple-darwin12 -filetype=obj -o %t.o < %s
; RUN: llvm-dwarfdump -debug-dump=pubnames %t.o | FileCheck --check-prefix=DARWIN %s
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

; Darwin shouldn't be generating the section by default
; DARWIN: debug_pubnames
; DARWIN: unit_size = 0x00000000

; Skip the output to the header of the pubnames section.
; LINUX: debug_pubnames

; Check for each name in the output.
; LINUX: global_namespace_variable
; LINUX: global_namespace_function
; LINUX: static_member_function
; LINUX: global_variable
; LINUX: global_function
; LINUX: member_function

%struct.C = type { i8 }

@_ZN1C22static_member_variableE = global i32 0, align 4
@global_variable = global %struct.C zeroinitializer, align 1
@_ZN2ns25global_namespace_variableE = global i32 1, align 4

define void @_ZN1C15member_functionEv(%struct.C* %this) nounwind uwtable align 2 {
entry:
  %this.addr = alloca %struct.C*, align 8
  store %struct.C* %this, %struct.C** %this.addr, align 8
  call void @llvm.dbg.declare(metadata !{%struct.C** %this.addr}, metadata !28), !dbg !30
  %this1 = load %struct.C** %this.addr
  store i32 0, i32* @_ZN1C22static_member_variableE, align 4, !dbg !31
  ret void, !dbg !32
}

declare void @llvm.dbg.declare(metadata, metadata) nounwind readnone

define i32 @_ZN1C22static_member_functionEv() nounwind uwtable align 2 {
entry:
  %0 = load i32* @_ZN1C22static_member_variableE, align 4, !dbg !33
  ret i32 %0, !dbg !33
}

define i32 @_Z15global_functionv() nounwind uwtable {
entry:
  ret i32 -1, !dbg !34
}

define void @_ZN2ns25global_namespace_functionEv() nounwind uwtable {
entry:
  call void @_ZN1C15member_functionEv(%struct.C* @global_variable), !dbg !35
  ret void, !dbg !36
}

attributes #0 = { nounwind uwtable }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!38}

!0 = metadata !{i32 786449, metadata !37, i32 4, metadata !"clang version 3.3 (http://llvm.org/git/clang.git a09cd8103a6a719cb2628cdf0c91682250a17bd2) (http://llvm.org/git/llvm.git 47d03cec0afca0c01ae42b82916d1d731716cd20)", i1 false, metadata !"", i32 0, metadata !1, metadata !1, metadata !2, metadata !24,  metadata !24, metadata !""} ; [ DW_TAG_compile_unit ] [/usr2/kparzysz/s.hex/t/dwarf-public-names.cpp] [DW_LANG_C_plus_plus]
!1 = metadata !{i32 0}
!2 = metadata !{metadata !3, metadata !18, metadata !19, metadata !20}
!3 = metadata !{i32 786478, metadata !4, null, metadata !"member_function", metadata !"member_function", metadata !"_ZN1C15member_functionEv", i32 9, metadata !5, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, void (%struct.C*)* @_ZN1C15member_functionEv, null, metadata !12, metadata !1, i32 9} ; [ DW_TAG_subprogram ] [line 9] [def] [member_function]
!4 = metadata !{i32 786473, metadata !37} ; [ DW_TAG_file_type ]
!5 = metadata !{i32 786453, i32 0, null, i32 0, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !6, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!6 = metadata !{null, metadata !7}
!7 = metadata !{i32 786447, i32 0, null, i32 0, i32 0, i64 64, i64 64, i64 0, i32 1088, metadata !8} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from C]
!8 = metadata !{i32 786451, metadata !37, null, metadata !"C", i32 1, i64 8, i64 8, i32 0, i32 0, null, metadata !9, i32 0, null, null, null} ; [ DW_TAG_structure_type ] [C] [line 1, size 8, align 8, offset 0] [def] [from ]
!9 = metadata !{metadata !10, metadata !12, metadata !14}
!10 = metadata !{i32 786445, metadata !37, metadata !8, metadata !"static_member_variable", i32 4, i64 0, i64 0, i64 0, i32 4096, metadata !11, null} ; [ DW_TAG_member ] [static_member_variable] [line 4, size 0, align 0, offset 0] [static] [from int]
!11 = metadata !{i32 786468, null, null, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!12 = metadata !{i32 786478, metadata !4, metadata !8, metadata !"member_function", metadata !"member_function", metadata !"_ZN1C15member_functionEv", i32 2, metadata !5, i1 false, i1 false, i32 0, i32 0, null, i32 256, i1 false, null, null, i32 0, metadata !13, i32 2} ; [ DW_TAG_subprogram ] [line 2] [member_function]
!13 = metadata !{i32 786468}                      ; [ DW_TAG_base_type ] [line 0, size 0, align 0, offset 0]
!14 = metadata !{i32 786478, metadata !4, metadata !8, metadata !"static_member_function", metadata !"static_member_function", metadata !"_ZN1C22static_member_functionEv", i32 3, metadata !15, i1 false, i1 false, i32 0, i32 0, null, i32 256, i1 false, null, null, i32 0, metadata !17, i32 3} ; [ DW_TAG_subprogram ] [line 3] [static_member_function]
!15 = metadata !{i32 786453, i32 0, null, i32 0, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !16, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!16 = metadata !{metadata !11}
!17 = metadata !{i32 786468}                      ; [ DW_TAG_base_type ] [line 0, size 0, align 0, offset 0]
!18 = metadata !{i32 786478, metadata !4, null, metadata !"static_member_function", metadata !"static_member_function", metadata !"_ZN1C22static_member_functionEv", i32 13, metadata !15, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, i32 ()* @_ZN1C22static_member_functionEv, null, metadata !14, metadata !1, i32 13} ; [ DW_TAG_subprogram ] [line 13] [def] [static_member_function]
!19 = metadata !{i32 786478, metadata !4, metadata !4, metadata !"global_function", metadata !"global_function", metadata !"_Z15global_functionv", i32 19, metadata !15, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, i32 ()* @_Z15global_functionv, null, null, metadata !1, i32 19} ; [ DW_TAG_subprogram ] [line 19] [def] [global_function]
!20 = metadata !{i32 786478, metadata !4, metadata !21, metadata !"global_namespace_function", metadata !"global_namespace_function", metadata !"_ZN2ns25global_namespace_functionEv", i32 24, metadata !22, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, void ()* @_ZN2ns25global_namespace_functionEv, null, null, metadata !1, i32 24} ; [ DW_TAG_subprogram ] [line 24] [def] [global_namespace_function]
!21 = metadata !{i32 786489, null, metadata !"ns", metadata !4, i32 23} ; [ DW_TAG_namespace ] [/usr2/kparzysz/s.hex/t/dwarf-public-names.cpp]
!22 = metadata !{i32 786453, i32 0, null, i32 0, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !23, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!23 = metadata !{null}
!24 = metadata !{metadata !25, metadata !26, metadata !27}
!25 = metadata !{i32 786484, i32 0, metadata !8, metadata !"static_member_variable", metadata !"static_member_variable", metadata !"_ZN1C22static_member_variableE", metadata !4, i32 7, metadata !11, i32 0, i32 1, i32* @_ZN1C22static_member_variableE, metadata !10} ; [ DW_TAG_variable ] [static_member_variable] [line 7] [def]
!26 = metadata !{i32 786484, i32 0, null, metadata !"global_variable", metadata !"global_variable", metadata !"", metadata !4, i32 17, metadata !8, i32 0, i32 1, %struct.C* @global_variable, null} ; [ DW_TAG_variable ] [global_variable] [line 17] [def]
!27 = metadata !{i32 786484, i32 0, metadata !21, metadata !"global_namespace_variable", metadata !"global_namespace_variable", metadata !"_ZN2ns25global_namespace_variableE", metadata !4, i32 27, metadata !11, i32 0, i32 1, i32* @_ZN2ns25global_namespace_variableE, null} ; [ DW_TAG_variable ] [global_namespace_variable] [line 27] [def]
!28 = metadata !{i32 786689, metadata !3, metadata !"this", metadata !4, i32 16777225, metadata !29, i32 1088, i32 0} ; [ DW_TAG_arg_variable ] [this] [line 9]
!29 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 0, metadata !8} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from C]
!30 = metadata !{i32 9, i32 0, metadata !3, null}
!31 = metadata !{i32 10, i32 0, metadata !3, null}
!32 = metadata !{i32 11, i32 0, metadata !3, null}
!33 = metadata !{i32 14, i32 0, metadata !18, null}
!34 = metadata !{i32 20, i32 0, metadata !19, null}
!35 = metadata !{i32 25, i32 0, metadata !20, null}
!36 = metadata !{i32 26, i32 0, metadata !20, null}
!37 = metadata !{metadata !"dwarf-public-names.cpp", metadata !"/usr2/kparzysz/s.hex/t"}
!38 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
