; RUN: llc %s -o - -filetype=asm -O0 -mtriple=x86_64-unknown-linux-gnu \
; RUN:   | FileCheck --check-prefix=CHECK --check-prefix=SINGLE --check-prefix=SINGLE-64 %s

; RUN: llc %s -o - -filetype=asm -O0 -mtriple=i386-linux-gnu \
; RUN:   | FileCheck --check-prefix=CHECK --check-prefix=SINGLE --check-prefix=SINGLE-32 %s

; RUN: llc %s -o - -filetype=asm -O0 -mtriple=x86_64-unknown-linux-gnu -split-dwarf=Enable \
; RUN:   | FileCheck --check-prefix=CHECK --check-prefix=FISSION %s

; FIXME: add relocation and DWARF expression support to llvm-dwarfdump & use
; that here instead of raw assembly printing

; FISSION: .section    .debug_info.dwo,
; 3 bytes of data in this DW_FORM_block1 representation of the location of 'tls'
; FISSION: .byte 3{{ *}}# DW_AT_location
; DW_OP_GNU_const_index (0xfx == 252) to refer to the debug_addr table
; FISSION-NEXT: .byte 252
; an index of zero into the debug_addr table
; FISSION-NEXT: .byte 0

; SINGLE: .section     .debug_info,
; 10 bytes of data in this DW_FORM_block1 representation of the location of 'tls'
; SINGLE-64: .byte     10 # DW_AT_location
; DW_OP_const8u (0x0e == 14) of address
; SINGLE-64-NEXT: .byte        14
; SINGLE-64-NEXT: .quad tls@DTPOFF

; SINGLE-32: .byte     6 # DW_AT_location
; DW_OP_const4u (0x0e == 12) of address
; SINGLE-32-NEXT: .byte        12
; SINGLE-32-NEXT: .long tls@DTPOFF

; DW_OP_GNU_push_tls_address
; CHECK-NEXT: .byte 224

; FISSION: DW_TAG_variable
; FISSION: .byte 2 # DW_AT_location
; DW_OP_GNU_addr_index
; FISSION-NEXT: .byte 251
; FISSION-NEXT: .byte 1

; FISSION: DW_TAG_template_value_parameter
; FISSION: .byte 3 # DW_AT_location
; DW_OP_GNU_addr_index
; FISSION-NEXT: .byte 251
; FISSION-NEXT: .byte 1
; DW_OP_stack_value
; FISSION-NEXT: .byte 159

; check that the expected TLS address description is the first thing in the debug_addr section
; FISSION: .section    .debug_addr
; FISSION-NEXT: .quad  tls@DTPOFF
; FISSION-NEXT: .quad  glbl
; FISSION-NOT: .quad  glbl

; Generated from:

; __thread int tls;
; int glbl;
;
; template <int *I>
; int func() {
;   return 0;
; }
;
; template int func<&glbl>(); // create a second reference to 'glbl'


@tls = thread_local global i32 0, align 4
@glbl = global i32 0, align 4

; Function Attrs: nounwind uwtable
define weak_odr i32 @_Z4funcIXadL_Z4glblEEEiv() #0 {
entry:
  ret i32 0, !dbg !18
}

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!15, !16}
!llvm.ident = !{!17}

!0 = !{!"0x11\004\00clang version 3.5 \000\00\000\00-.dwo\000", !1, !2, !2, !3, !12, !2} ; [ DW_TAG_compile_unit ] [/tmp/dbginfo/tls.cpp] [DW_LANG_C_plus_plus]
!1 = !{!"tls.cpp", !"/tmp/dbginfo"}
!2 = !{}
!3 = !{!4}
!4 = !{!"0x2e\00func<&glbl>\00func<&glbl>\00_Z4funcIXadL_Z4glblEEEiv\005\000\001\000\006\00256\000\005", !1, !5, !6, null, i32 ()* @_Z4funcIXadL_Z4glblEEEiv, !9, null, !2} ; [ DW_TAG_subprogram ] [line 5] [def] [func<&glbl>]
!5 = !{!"0x29", !1}          ; [ DW_TAG_file_type ] [/tmp/dbginfo/tls.cpp]
!6 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !7, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = !{!8}
!8 = !{!"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!9 = !{!10}
!10 = !{!"0x30\00I\000\000", null, !11, i32* @glbl, null} ; [ DW_TAG_template_value_parameter ]
!11 = !{!"0xf\00\000\0064\0064\000\000", null, null, !8} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from int]
!12 = !{!13, !14}
!13 = !{!"0x34\00tls\00tls\00\001\000\001", null, !5, !8, i32* @tls, null} ; [ DW_TAG_variable ] [tls] [line 1] [def]
!14 = !{!"0x34\00glbl\00glbl\00\002\000\001", null, !5, !8, i32* @glbl, null} ; [ DW_TAG_variable ] [glbl] [line 2] [def]
!15 = !{i32 2, !"Dwarf Version", i32 4}
!16 = !{i32 1, !"Debug Info Version", i32 2}
!17 = !{!"clang version 3.5 "}
!18 = !MDLocation(line: 6, scope: !4)
