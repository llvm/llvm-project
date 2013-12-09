; RUN: llc -mtriple=x86_64-pc-linux-gnu %s -o %t -filetype=obj
; RUN: llvm-dwarfdump -debug-dump=info %t | FileCheck %s

; ModuleID = 'test.c'

@GLB = common global i32 0, align 4

define i32 @f() nounwind {
  %LOC = alloca i32, align 4
  call void @llvm.dbg.declare(metadata !{i32* %LOC}, metadata !15), !dbg !17
  %1 = load i32* @GLB, align 4, !dbg !18
  store i32 %1, i32* %LOC, align 4, !dbg !18
  %2 = load i32* @GLB, align 4, !dbg !19
  ret i32 %2, !dbg !19
}

declare void @llvm.dbg.declare(metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!21}

!0 = metadata !{i32 786449, metadata !20, i32 12, metadata !"clang version 3.0 (trunk)", i1 false, metadata !"", i32 0, metadata !1, metadata !1, metadata !3, metadata !12,  metadata !12, metadata !""} ; [ DW_TAG_compile_unit ]
!1 = metadata !{i32 0}
!3 = metadata !{metadata !5}
!5 = metadata !{i32 720942, metadata !6, metadata !6, metadata !"f", metadata !"f", metadata !"", i32 3, metadata !7, i1 false, i1 true, i32 0, i32 0, null, i32 0, i1 false, i32 ()* @f, null, null, metadata !10, i32 0} ; [ DW_TAG_subprogram ] [line 3] [def] [scope 0] [f]
!6 = metadata !{i32 720937, metadata !20} ; [ DW_TAG_file_type ]
!7 = metadata !{i32 720917, i32 0, null, i32 0, i32 0, i64 0, i64 0, i32 0, i32 0, null, metadata !8, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = metadata !{metadata !9}
!9 = metadata !{i32 720932, null, null, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
!10 = metadata !{metadata !11}
!11 = metadata !{i32 720932}                      ; [ DW_TAG_base_type ]
!12 = metadata !{metadata !14}
!14 = metadata !{i32 720948, i32 0, null, metadata !"GLB", metadata !"GLB", metadata !"", metadata !6, i32 1, metadata !9, i32 0, i32 1, i32* @GLB, null} ; [ DW_TAG_variable ]
!15 = metadata !{i32 786688, metadata !16, metadata !"LOC", metadata !6, i32 4, metadata !9, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!16 = metadata !{i32 786443, metadata !20, metadata !5, i32 3, i32 9, i32 0} ; [ DW_TAG_lexical_block ]
!17 = metadata !{i32 4, i32 9, metadata !16, null}
!18 = metadata !{i32 4, i32 23, metadata !16, null}
!19 = metadata !{i32 5, i32 5, metadata !16, null}
!20 = metadata !{metadata !"test.c", metadata !"/work/llvm/vanilla/test/DebugInfo"}

; CHECK: DW_TAG_variable
; CHECK-NEXT: DW_AT_name [DW_FORM_strp]       ( .debug_str[0x{{[0-9a-f]*}}] = "GLB")
; CHECK: DW_AT_decl_file [DW_FORM_data1] (0x01)
; CHECK: DW_AT_decl_line [DW_FORM_data1] (0x01)

; CHECK: DW_TAG_variable
; CHECK-NEXT: DW_AT_name [DW_FORM_strp]   ( .debug_str[0x{{[0-9a-f]*}}] = "LOC")
; CHECK: DW_AT_decl_file [DW_FORM_data1]     (0x01)
; CHECK: DW_AT_decl_line [DW_FORM_data1]     (0x04)

!21 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
