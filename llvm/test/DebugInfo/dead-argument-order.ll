; REQUIRES: object-emission

; RUN: %llc_dwarf -O0 -filetype=obj < %s | llvm-dwarfdump -debug-dump=info - | FileCheck %s

; Built from the following source with clang -O1
; struct S { int i; };
; int function(struct S s, int i) { return s.i + i; }

; Due to the X86_64 ABI, 's' is passed in registers and once optimized, the
; entirety of 's' is never reconstituted, since only the int is required, and
; thus the variable's location is unknown/dead to debug info.

; Future/current work should enable us to describe partial variables, which, in
; this case, happens to be the entire variable.

; CHECK: DW_TAG_subprogram
; CHECK-NOT: DW_TAG
; CHECK:   DW_AT_name {{.*}} "function"
; CHECK-NOT: {{DW_TAG|NULL}}
; CHECK:   DW_TAG_formal_parameter
; CHECK-NOT: DW_TAG
; CHECK:     DW_AT_name {{.*}} "s"
; CHECK-NOT: DW_TAG
; FIXME: Even though 's' is never reconstituted into a struct, the one member
; variable is still live and used, and so we should be able to describe 's's
; location as the location of that int.
; CHECK-NOT: DW_AT_location
; CHECK-NOT: {{DW_TAG|NULL}}
; CHECK:   DW_TAG_formal_parameter
; CHECK-NOT: DW_TAG
; CHECK:     DW_AT_location
; CHECK-NOT: DW_TAG
; CHECK:     DW_AT_name {{.*}} "i"


%struct.S = type { i32 }

; Function Attrs: nounwind readnone uwtable
define i32 @_Z8function1Si(i32 %s.coerce, i32 %i) #0 {
entry:
  tail call void @llvm.dbg.declare(metadata %struct.S* undef, metadata !14, metadata !{!"0x102"}), !dbg !20
  tail call void @llvm.dbg.value(metadata i32 %i, i64 0, metadata !15, metadata !{!"0x102"}), !dbg !20
  %add = add nsw i32 %i, %s.coerce, !dbg !20
  ret i32 %add, !dbg !20
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #1

attributes #0 = { nounwind readnone uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!16, !17}
!llvm.ident = !{!18}

!0 = !{!"0x11\004\00clang version 3.5.0 \001\00\000\00\001", !1, !2, !3, !8, !2, !2} ; [ DW_TAG_compile_unit ] [/tmp/dbginfo/dead-argument-order.cpp] [DW_LANG_C_plus_plus]
!1 = !{!"dead-argument-order.cpp", !"/tmp/dbginfo"}
!2 = !{}
!3 = !{!4}
!4 = !{!"0x13\00S\001\0032\0032\000\000\000", !1, null, null, !5, null, null, !"_ZTS1S"} ; [ DW_TAG_structure_type ] [S] [line 1, size 32, align 32, offset 0] [def] [from ]
!5 = !{!6}
!6 = !{!"0xd\00i\001\0032\0032\000\000", !1, !"_ZTS1S", !7} ; [ DW_TAG_member ] [i] [line 1, size 32, align 32, offset 0] [from int]
!7 = !{!"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!8 = !{!9}
!9 = !{!"0x2e\00function\00function\00_Z8function1Si\002\000\001\000\006\00256\001\002", !1, !10, !11, null, i32 (i32, i32)* @_Z8function1Si, null, null, !13} ; [ DW_TAG_subprogram ] [line 2] [def] [function]
!10 = !{!"0x29", !1}         ; [ DW_TAG_file_type ] [/tmp/dbginfo/dead-argument-order.cpp]
!11 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !12, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!12 = !{!7, !4, !7}
!13 = !{!14, !15}
!14 = !{!"0x101\00s\0016777218\000", !9, !10, !"_ZTS1S"} ; [ DW_TAG_arg_variable ] [s] [line 2]
!15 = !{!"0x101\00i\0033554434\000", !9, !10, !7} ; [ DW_TAG_arg_variable ] [i] [line 2]
!16 = !{i32 2, !"Dwarf Version", i32 4}
!17 = !{i32 2, !"Debug Info Version", i32 2}
!18 = !{!"clang version 3.5.0 "}
!19 = !{%struct.S* undef}
!20 = !MDLocation(line: 2, scope: !9)

