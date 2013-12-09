; RUN: llc -mtriple=x86_64-apple-darwin -O0 -filetype=obj -o %t < %s
; RUN: llvm-dwarfdump -debug-dump=info %t | FileCheck %s
; <rdar://problem/12566646>

%struct.foo = type { i32, [1 x i32] }
%struct.bar = type { i32, [0 x i32] }

define i32 @func() nounwind uwtable ssp {
entry:
  %my_foo = alloca %struct.foo, align 4
  %my_bar = alloca %struct.bar, align 4
  call void @llvm.dbg.declare(metadata !{%struct.foo* %my_foo}, metadata !10), !dbg !19
  call void @llvm.dbg.declare(metadata !{%struct.bar* %my_bar}, metadata !20), !dbg !28
  %a = getelementptr inbounds %struct.foo* %my_foo, i32 0, i32 0, !dbg !29
  store i32 3, i32* %a, align 4, !dbg !29
  %a1 = getelementptr inbounds %struct.bar* %my_bar, i32 0, i32 0, !dbg !30
  store i32 5, i32* %a1, align 4, !dbg !30
  %a2 = getelementptr inbounds %struct.foo* %my_foo, i32 0, i32 0, !dbg !31
  %0 = load i32* %a2, align 4, !dbg !31
  %a3 = getelementptr inbounds %struct.bar* %my_bar, i32 0, i32 0, !dbg !31
  %1 = load i32* %a3, align 4, !dbg !31
  %add = add nsw i32 %0, %1, !dbg !31
  ret i32 %add, !dbg !31
}

declare void @llvm.dbg.declare(metadata, metadata) nounwind readnone

; An empty array should not have an AT_upper_bound attribute. But an array of 1
; should.

; CHECK:      DW_TAG_base_type
; CHECK-NEXT: DW_AT_name [DW_FORM_strp]  ( .debug_str[{{.*}}] = "int")
; CHECK-NEXT: DW_AT_encoding [DW_FORM_data1]   (0x05)
; CHECK-NEXT: DW_AT_byte_size [DW_FORM_data1]  (0x04)

; int foo::b[1]:
; CHECK: DW_TAG_structure_type
; CHECK: DW_AT_name{{.*}}"foo"
; CHECK:      DW_TAG_member
; CHECK:      DW_TAG_member
; CHECK-NEXT: DW_AT_name [DW_FORM_strp]  ( .debug_str[{{.*}}] = "b")
; CHECK-NEXT: DW_AT_type [DW_FORM_ref4]

; int[1]:
; CHECK:      DW_TAG_array_type [{{.*}}] *
; CHECK-NEXT: DW_AT_type [DW_FORM_ref4]
; CHECK:      DW_TAG_subrange_type [{{.*}}]
; CHECK-NEXT: DW_AT_type [DW_FORM_ref4]
; CHECK-NEXT: DW_AT_upper_bound [DW_FORM_data1]  (0x00)

; int bar::b[0]:
; CHECK: DW_TAG_structure_type
; CHECK: DW_AT_name{{.*}}"bar"
; CHECK:      DW_TAG_member
; CHECK:      DW_TAG_member
; CHECK-NEXT: DW_AT_name [DW_FORM_strp]  ( .debug_str[{{.*}}] = "b")
; CHECK-NEXT: DW_AT_type [DW_FORM_ref4]

; int[0]:
; CHECK:      DW_TAG_array_type [{{.*}}] *
; CHECK-NEXT: DW_AT_type [DW_FORM_ref4]
; CHECK:      DW_TAG_subrange_type [11]
; CHECK-NEXT: DW_AT_type [DW_FORM_ref4]
; CHECK-NOT:  DW_AT_upper_bound

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!33}

!0 = metadata !{i32 786449, metadata !32, i32 12, metadata !"clang version 3.3 (trunk 169136)", i1 false, metadata !"", i32 0, metadata !1, metadata !1, metadata !3, metadata !1,  metadata !1, metadata !""} ; [ DW_TAG_compile_unit ] [/Volumes/Sandbox/llvm/test.c] [DW_LANG_C99]
!1 = metadata !{i32 0}
!3 = metadata !{metadata !5}
!5 = metadata !{i32 786478, metadata !6, metadata !6, metadata !"func", metadata !"func", metadata !"", i32 11, metadata !7, i1 false, i1 true, i32 0, i32 0, null, i32 0, i1 false, i32 ()* @func, null, null, metadata !1, i32 11} ; [ DW_TAG_subprogram ] [line 11] [def] [func]
!6 = metadata !{i32 786473, metadata !32} ; [ DW_TAG_file_type ]
!7 = metadata !{i32 786453, i32 0, null, i32 0, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !8, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = metadata !{metadata !9}
!9 = metadata !{i32 786468, null, null, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!10 = metadata !{i32 786688, metadata !11, metadata !"my_foo", metadata !6, i32 12, metadata !12, i32 0, i32 0} ; [ DW_TAG_auto_variable ] [my_foo] [line 12]
!11 = metadata !{i32 786443, metadata !6, metadata !5, i32 11, i32 0, i32 0} ; [ DW_TAG_lexical_block ] [/Volumes/Sandbox/llvm/test.c]
!12 = metadata !{i32 786451, metadata !32, null, metadata !"foo", i32 1, i64 64, i64 32, i32 0, i32 0, null, metadata !13, i32 0, null, i32 0, null} ; [ DW_TAG_structure_type ] [foo] [line 1, size 64, align 32, offset 0] [def] [from ]
!13 = metadata !{metadata !14, metadata !15}
!14 = metadata !{i32 786445, metadata !32, metadata !12, metadata !"a", i32 2, i64 32, i64 32, i64 0, i32 0, metadata !9} ; [ DW_TAG_member ] [a] [line 2, size 32, align 32, offset 0] [from int]
!15 = metadata !{i32 786445, metadata !32, metadata !12, metadata !"b", i32 3, i64 32, i64 32, i64 32, i32 0, metadata !16} ; [ DW_TAG_member ] [b] [line 3, size 32, align 32, offset 32] [from ]
!16 = metadata !{i32 786433, null, null, metadata !"", i32 0, i64 32, i64 32, i32 0, i32 0, metadata !9, metadata !17, i32 0, null, null, null} ; [ DW_TAG_array_type ] [line 0, size 32, align 32, offset 0] [from int]
!17 = metadata !{metadata !18}
!18 = metadata !{i32 786465, i64 0, i64 1} ; [ DW_TAG_subrange_type ] [0, 1]
!19 = metadata !{i32 12, i32 0, metadata !11, null}
!20 = metadata !{i32 786688, metadata !11, metadata !"my_bar", metadata !6, i32 13, metadata !21, i32 0, i32 0} ; [ DW_TAG_auto_variable ] [my_bar] [line 13]
!21 = metadata !{i32 786451, metadata !32, null, metadata !"bar", i32 6, i64 32, i64 32, i32 0, i32 0, null, metadata !22, i32 0, null, i32 0, null} ; [ DW_TAG_structure_type ] [bar] [line 6, size 32, align 32, offset 0] [def] [from ]
!22 = metadata !{metadata !23, metadata !24}
!23 = metadata !{i32 786445, metadata !32, metadata !21, metadata !"a", i32 7, i64 32, i64 32, i64 0, i32 0, metadata !9} ; [ DW_TAG_member ] [a] [line 7, size 32, align 32, offset 0] [from int]
!24 = metadata !{i32 786445, metadata !32, metadata !21, metadata !"b", i32 8, i64 0, i64 32, i64 32, i32 0, metadata !25} ; [ DW_TAG_member ] [b] [line 8, size 0, align 32, offset 32] [from ]
!25 = metadata !{i32 786433, null, null, metadata !"", i32 0, i64 0, i64 32, i32 0, i32 0, metadata !9, metadata !26, i32 0, null, null, null} ; [ DW_TAG_array_type ] [line 0, size 0, align 32, offset 0] [from int]
!26 = metadata !{metadata !27}
!27 = metadata !{i32 786465, i64 0, i64 0} ; [ DW_TAG_subrange_type ] [0, 0]
!28 = metadata !{i32 13, i32 0, metadata !11, null}
!29 = metadata !{i32 15, i32 0, metadata !11, null}
!30 = metadata !{i32 16, i32 0, metadata !11, null}
!31 = metadata !{i32 17, i32 0, metadata !11, null}
!32 = metadata !{metadata !"test.c", metadata !"/Volumes/Sandbox/llvm"}
!33 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
