; RUN: llc -mtriple=x86_64-apple-darwin -O0 -filetype=obj -o %t < %s
; RUN: llvm-dwarfdump -debug-dump=info %t | FileCheck %s
; <rdar://problem/12566646>

%struct.foo = type { i32, [1 x i32] }
%struct.bar = type { i32, [0 x i32] }

define i32 @func() nounwind uwtable ssp {
entry:
  %my_foo = alloca %struct.foo, align 4
  %my_bar = alloca %struct.bar, align 4
  call void @llvm.dbg.declare(metadata %struct.foo* %my_foo, metadata !10, metadata !{!"0x102"}), !dbg !19
  call void @llvm.dbg.declare(metadata %struct.bar* %my_bar, metadata !20, metadata !{!"0x102"}), !dbg !28
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

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

; CHECK:      DW_TAG_base_type
; CHECK-NEXT: DW_AT_name [DW_FORM_strp]  ( .debug_str[{{.*}}] = "int")
; CHECK-NEXT: DW_AT_encoding [DW_FORM_data1]   (DW_ATE_signed)
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
; CHECK-NEXT: DW_AT_count [DW_FORM_data1]  (0x01)

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
; CHECK:      DW_TAG_subrange_type
; CHECK-NEXT: DW_AT_type [DW_FORM_ref4]
; CHECK:      DW_AT_count [DW_FORM_data1]  (0x00)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!33}

!0 = !{!"0x11\0012\00clang version 3.3 (trunk 169136)\000\00\000\00\000", !32, !1, !1, !3, !1,  !1} ; [ DW_TAG_compile_unit ] [/Volumes/Sandbox/llvm/test.c] [DW_LANG_C99]
!1 = !{}
!3 = !{!5}
!5 = !{!"0x2e\00func\00func\00\0011\000\001\000\006\000\000\0011", !6, !6, !7, null, i32 ()* @func, null, null, !1} ; [ DW_TAG_subprogram ] [line 11] [def] [func]
!6 = !{!"0x29", !32} ; [ DW_TAG_file_type ]
!7 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !8, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = !{!9}
!9 = !{!"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!10 = !{!"0x100\00my_foo\0012\000", !11, !6, !12} ; [ DW_TAG_auto_variable ] [my_foo] [line 12]
!11 = !{!"0xb\0011\000\000", !6, !5} ; [ DW_TAG_lexical_block ] [/Volumes/Sandbox/llvm/test.c]
!12 = !{!"0x13\00foo\001\0064\0032\000\000\000", !32, null, null, !13, null, i32 0, null} ; [ DW_TAG_structure_type ] [foo] [line 1, size 64, align 32, offset 0] [def] [from ]
!13 = !{!14, !15}
!14 = !{!"0xd\00a\002\0032\0032\000\000", !32, !12, !9} ; [ DW_TAG_member ] [a] [line 2, size 32, align 32, offset 0] [from int]
!15 = !{!"0xd\00b\003\0032\0032\0032\000", !32, !12, !16} ; [ DW_TAG_member ] [b] [line 3, size 32, align 32, offset 32] [from ]
!16 = !{!"0x1\00\000\0032\0032\000\000", null, null, !9, !17, i32 0, null, null, null} ; [ DW_TAG_array_type ] [line 0, size 32, align 32, offset 0] [from int]
!17 = !{!18}
!18 = !{!"0x21\000\001"} ; [ DW_TAG_subrange_type ] [0, 1]
!19 = !MDLocation(line: 12, scope: !11)
!20 = !{!"0x100\00my_bar\0013\000", !11, !6, !21} ; [ DW_TAG_auto_variable ] [my_bar] [line 13]
!21 = !{!"0x13\00bar\006\0032\0032\000\000\000", !32, null, null, !22, null, i32 0, null} ; [ DW_TAG_structure_type ] [bar] [line 6, size 32, align 32, offset 0] [def] [from ]
!22 = !{!23, !24}
!23 = !{!"0xd\00a\007\0032\0032\000\000", !32, !21, !9} ; [ DW_TAG_member ] [a] [line 7, size 32, align 32, offset 0] [from int]
!24 = !{!"0xd\00b\008\000\0032\0032\000", !32, !21, !25} ; [ DW_TAG_member ] [b] [line 8, size 0, align 32, offset 32] [from ]
!25 = !{!"0x1\00\000\000\0032\000\000", null, null, !9, !26, i32 0, null, null, null} ; [ DW_TAG_array_type ] [line 0, size 0, align 32, offset 0] [from int]
!26 = !{!27}
!27 = !{!"0x21\000\000"} ; [ DW_TAG_subrange_type ] [0, 0]
!28 = !MDLocation(line: 13, scope: !11)
!29 = !MDLocation(line: 15, scope: !11)
!30 = !MDLocation(line: 16, scope: !11)
!31 = !MDLocation(line: 17, scope: !11)
!32 = !{!"test.c", !"/Volumes/Sandbox/llvm"}
!33 = !{i32 1, !"Debug Info Version", i32 2}
