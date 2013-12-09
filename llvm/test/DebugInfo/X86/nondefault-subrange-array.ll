; RUN: llc -mtriple=x86_64-apple-darwin -O0 -filetype=obj -o %t < %s
; RUN: llvm-dwarfdump -debug-dump=info %t | FileCheck %s

%class.A = type { [42 x i32] }

@a = global %class.A zeroinitializer, align 4

; Check that we can handle non-default array bounds. In this case, the array
; goes from [-3, 38].

; CHECK: DW_TAG_class_type
; CHECK: DW_TAG_member
; CHECK-NEXT:                   DW_AT_name [DW_FORM_strp]       ( .debug_str[0x{{[0-9a-f]*}}] = "x")
; CHECK-NEXT:                   DW_AT_type [DW_FORM_ref4]       (cu + 0x{{[0-9a-f]*}} => {[[ARRAY:0x[0-9a-f]*]]})

; CHECK: [[ARRAY]]: DW_TAG_array_type [{{.*}}] *
; CHECK-NEXT:                 DW_AT_type [DW_FORM_ref4]    (cu + 0x{{[0-9a-f]*}} => {[[BASE:0x[0-9a-f]*]]})

; CHECK: DW_TAG_subrange_type
; CHECK-NEXT:                   DW_AT_type [DW_FORM_ref4]  (cu + 0x{{[0-9a-f]*}} => {[[BASE2:0x[0-9a-f]*]]})
; CHECK-NEXT:                   DW_AT_lower_bound [DW_FORM_data8]       (0xfffffffffffffffd)
; CHECK-NEXT:                   DW_AT_upper_bound [DW_FORM_data1]       (0x26)

; CHECK: [[BASE]]: DW_TAG_base_type
; CHECK: [[BASE2]]: DW_TAG_base_type
; CHECK-NEXT:                 DW_AT_name [DW_FORM_strp]       ( .debug_str[0x{{[0-9a-f]*}}] = "int")
; CHECK-NEXT:                 DW_AT_byte_size [DW_FORM_data1] (0x04)
; CHECK-NEXT:                 DW_AT_encoding [DW_FORM_data1]  (0x05)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!21}

!0 = metadata !{i32 786449, metadata !20, i32 4, metadata !"clang version 3.3 (trunk 169136)", i1 false, metadata !"", i32 0, metadata !1, metadata !1, metadata !1, metadata !3,  metadata !3, metadata !""} ; [ DW_TAG_compile_unit ] [/Volumes/Sandbox/llvm/t.cpp] [DW_LANG_C_plus_plus]
!1 = metadata !{i32 0}
!3 = metadata !{metadata !5}
!5 = metadata !{i32 786484, i32 0, null, metadata !"a", metadata !"a", metadata !"", metadata !6, i32 1, metadata !7, i32 0, i32 1, %class.A* @a, null} ; [ DW_TAG_variable ] [a] [line 1] [def]
!6 = metadata !{i32 786473, metadata !20} ; [ DW_TAG_file_type ]
!7 = metadata !{i32 786434, metadata !20, null, metadata !"A", i32 1, i64 0, i64 32, i32 0, i32 0, null, metadata !8, i32 0, null, null, null} ; [ DW_TAG_class_type ] [A] [line 1, size 0, align 32, offset 0] [def] [from ]
!8 = metadata !{metadata !9, metadata !14}
!9 = metadata !{i32 786445, metadata !20, metadata !7, metadata !"x", i32 1, i64 0, i64 0, i64 0, i32 1, metadata !10} ; [ DW_TAG_member ] [x] [line 1, size 0, align 0, offset 0] [private] [from ]
!10 = metadata !{i32 786433, null, null, metadata !"", i32 0, i64 0, i64 32, i32 0, i32 0, metadata !11, metadata !12, i32 0, null, null, null} ; [ DW_TAG_array_type ] [line 0, size 0, align 32, offset 0] [from int]
!11 = metadata !{i32 786468, null, null, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!12 = metadata !{metadata !13}
!13 = metadata !{i32 786465, i64 -3, i64 42} ; [ DW_TAG_subrange_type ] [-3, 39]
!14 = metadata !{i32 786478, metadata !6, metadata !7, metadata !"A", metadata !"A", metadata !"", i32 1, metadata !15, i1 false, i1 false, i32 0, i32 0, null, i32 320, i1 false, null, null, i32 0, metadata !18, i32 1} ; [ DW_TAG_subprogram ] [line 1] [A]
!15 = metadata !{i32 786453, i32 0, null, i32 0, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !16, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!16 = metadata !{null, metadata !17}
!17 = metadata !{i32 786447, i32 0, null, i32 0, i32 0, i64 64, i64 64, i64 0, i32 1088, metadata !7} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from A]
!18 = metadata !{metadata !19}
!19 = metadata !{i32 786468}                      ; [ DW_TAG_base_type ] [line 0, size 0, align 0, offset 0]
!20 = metadata !{metadata !"t.cpp", metadata !"/Volumes/Sandbox/llvm"}
!21 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
