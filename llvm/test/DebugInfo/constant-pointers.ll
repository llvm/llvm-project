; REQUIRES: object-emission

; RUN: %llc_dwarf -O0 -filetype=obj %s -o - | llvm-dwarfdump -debug-dump=info - | FileCheck %s

; Ensure that pointer constants are emitted as unsigned data. Alternatively,
; these could be signless data (dataN).

; Built with Clang from:
; template <void *V, void (*F)(), int i>
; void func() {}
; template void func<nullptr, nullptr, 42>();

; CHECK: DW_TAG_subprogram
; CHECK:   DW_TAG_template_value_parameter
; CHECK:     DW_AT_name {{.*}} "V"
; CHECK:     DW_AT_const_value [DW_FORM_udata] (0)
; CHECK:   DW_TAG_template_value_parameter
; CHECK:     DW_AT_name {{.*}} "F"
; CHECK:     DW_AT_const_value [DW_FORM_udata] (0)

; Function Attrs: nounwind uwtable
define weak_odr void @_Z4funcILPv0ELPFvvE0ELi42EEvv() #0 {
entry:
  ret void, !dbg !18
}

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!15, !16}
!llvm.ident = !{!17}

!0 = !{!"0x11\004\00clang version 3.5.0 \000\00\000\00\001", !1, !2, !2, !3, !2, !2} ; [ DW_TAG_compile_unit ] [/tmp/dbginfo/constant-pointers.cpp] [DW_LANG_C_plus_plus]
!1 = !{!"constant-pointers.cpp", !"/tmp/dbginfo"}
!2 = !{}
!3 = !{!4}
!4 = !{!"0x2e\00func<nullptr, nullptr, 42>\00func<nullptr, nullptr, 42>\00_Z4funcILPv0ELPFvvE0ELi42EEvv\002\000\001\000\006\00256\000\002", !1, !5, !6, null, void ()* @_Z4funcILPv0ELPFvvE0ELi42EEvv, !8, null, !2} ; [ DW_TAG_subprogram ] [line 2] [def] [func<nullptr, nullptr, 42>]
!5 = !{!"0x29", !1}          ; [ DW_TAG_file_type ] [/tmp/dbginfo/constant-pointers.cpp]
!6 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !7, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = !{null}
!8 = !{!9, !11, !13}
!9 = !{!"0x30\00V\000\000", null, !10, i8 0, null} ; [ DW_TAG_template_value_parameter ]
!10 = !{!"0xf\00\000\0064\0064\000\000", null, null, null} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from ]
!11 = !{!"0x30\00F\000\000", null, !12, i8 0, null} ; [ DW_TAG_template_value_parameter ]
!12 = !{!"0xf\00\000\0064\0064\000\000", null, null, !6} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from ]
!13 = !{!"0x30\00i\000\000", null, !14, i32 42, null} ; [ DW_TAG_template_value_parameter ]
!14 = !{!"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!15 = !{i32 2, !"Dwarf Version", i32 4}
!16 = !{i32 2, !"Debug Info Version", i32 2}
!17 = !{!"clang version 3.5.0 "}
!18 = !MDLocation(line: 3, scope: !4)
