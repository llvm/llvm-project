; RUN: llc -mtriple=x86_64-pc-linux-gnu %s -o %t -filetype=obj
; RUN: llvm-dwarfdump -debug-dump=info %t | FileCheck %s

; ModuleID = 'test.c'

@GLB = common global i32 0, align 4

define i32 @f() nounwind {
  %LOC = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i32* %LOC, metadata !15, metadata !{!"0x102"}), !dbg !17
  %1 = load i32* @GLB, align 4, !dbg !18
  store i32 %1, i32* %LOC, align 4, !dbg !18
  %2 = load i32* @GLB, align 4, !dbg !19
  ret i32 %2, !dbg !19
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!21}

!0 = !{!"0x11\0012\00clang version 3.0 (trunk)\000\00\000\00\000", !20, !1, !1, !3, !12,  !1} ; [ DW_TAG_compile_unit ]
!1 = !{}
!3 = !{!5}
!5 = !{!"0x2e\00f\00f\00\003\000\001\000\006\000\000\000", !6, !6, !7, null, i32 ()* @f, null, null, null} ; [ DW_TAG_subprogram ] [line 3] [def] [scope 0] [f]
!6 = !{!"0x29", !20} ; [ DW_TAG_file_type ]
!7 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !8, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = !{!9}
!9 = !{!"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ]
!12 = !{!14}
!14 = !{!"0x34\00GLB\00GLB\00\001\000\001", null, !6, !9, i32* @GLB, null} ; [ DW_TAG_variable ]
!15 = !{!"0x100\00LOC\004\000", !16, !6, !9} ; [ DW_TAG_auto_variable ]
!16 = !{!"0xb\003\009\000", !20, !5} ; [ DW_TAG_lexical_block ]
!17 = !MDLocation(line: 4, column: 9, scope: !16)
!18 = !MDLocation(line: 4, column: 23, scope: !16)
!19 = !MDLocation(line: 5, column: 5, scope: !16)
!20 = !{!"test.c", !"/work/llvm/vanilla/test/DebugInfo"}

; CHECK: DW_TAG_variable
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_name [DW_FORM_strp]       ( .debug_str[0x{{[0-9a-f]*}}] = "GLB")
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_decl_file [DW_FORM_data1] ("/work/llvm/vanilla/test/DebugInfo{{[/\\]}}test.c")
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_decl_line [DW_FORM_data1] (1)

; CHECK: DW_TAG_variable
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_name [DW_FORM_strp]   ( .debug_str[0x{{[0-9a-f]*}}] = "LOC")
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_decl_file [DW_FORM_data1]     ("/work/llvm/vanilla/test/DebugInfo{{[/\\]}}test.c")
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_decl_line [DW_FORM_data1]     (4)

!21 = !{i32 1, !"Debug Info Version", i32 2}
