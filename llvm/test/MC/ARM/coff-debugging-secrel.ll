; RUN: llc -mtriple thumbv7--windows-itanium -filetype obj -o - %s \
; RUN:     | llvm-readobj -r - | FileCheck %s -check-prefix CHECK-ITANIUM

; RUN: llc -mtriple thumbv7--windows-msvc -filetype obj -o - %s \
; RUN:    | llvm-readobj -r - | FileCheck %s -check-prefix CHECK-MSVC

; ModuleID = '/Users/compnerd/work/llvm/test/MC/ARM/reduced.c'
target datalayout = "e-m:e-p:32:32-i1:8:32-i8:8:32-i16:16:32-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "thumbv7--windows-itanium"

define arm_aapcs_vfpcc void @function() {
entry:
  ret void, !dbg !0
}

!llvm.dbg.cu = !{!7}
!llvm.module.flags = !{!9, !10}

!0 = !MDLocation(line: 1, scope: !1)
!1 = !{!"0x2e\00function\00function\00\001\000\001\000\006\000\000\001", !2, !3, !4, null, void ()* @function, null, null, !6} ; [ DW_TAG_subprogram ], [line 1], [def], [function]
!2 = !{!"/Users/compnerd/work/llvm/test/MC/ARM/reduced.c", !"/Users/compnerd/work/llvm"}
!3 = !{!"0x29", !2} ; [ DW_TAG_file_type] [/Users/compnerd/work/llvm/test/MC/ARM/reduced.c]
!4 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !5, null, null, null} ; [ DW_TAG_subroutine_type ], [line 0, size 0, align 0, offset 0] [from ]
!5 = !{null}
!6 = !{}
!7 = !{!"0x11\0012\00clang version 3.5.0\000\00\000\00\001", !2, !6, !6, !8, !6, !6} ; [ DW_TAG_compile_unit ] [/Users/compnerd/work/llvm/test/MC/ARM/reduced.c] [DW_LANG_C99]
!8 = !{!1}
!9 = !{i32 2, !"Dwarf Version", i32 4}
!10 = !{i32 1, !"Debug Info Version", i32 2}

; CHECK-ITANIUM: Relocations [
; CHECK-ITANIUM:   Section {{.*}} .debug_info {
; CHECK-ITANIUM:     0x6 IMAGE_REL_ARM_SECREL .debug_abbrev
; CHECK-ITANIUM:     0xC IMAGE_REL_ARM_SECREL .debug_str
; CHECK-ITANIUM:     0x12 IMAGE_REL_ARM_SECREL .debug_str
; CHECK-ITANIUM:     0x16 IMAGE_REL_ARM_SECREL .debug_line
; CHECK-ITANIUM:   }
; CHECK-ITANIUM:   Section {{.*}}.debug_pubnames {
; CHECK-ITANIUM:     0x6 IMAGE_REL_ARM_SECREL .debug_info
; CHECK-ITANIUM:   }
; CHECK-ITANIUM: ]

; CHECK-MSVC: Relocations [
; CHECK-MSVC:   Section {{.*}} .debug$S {
; CHECK-MSVC:     0x2C IMAGE_REL_ARM_SECREL function
; CHECK-MSVC:     0x30 IMAGE_REL_ARM_SECTION function
; CHECK-MSVC:     0x48 IMAGE_REL_ARM_SECREL function
; CHECK-MSVC:     0x4C IMAGE_REL_ARM_SECTION function
; CHECK-MSVC:   }
; CHECK-MSVC: ]

