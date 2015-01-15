; REQUIRES: object-emission

; RUN: llc -mtriple=x86_64-linux -O0 -filetype=obj < %s \
; RUN:     | llvm-dwarfdump -debug-dump=info - | FileCheck --check-prefix=CHECK --check-prefix=CHECK-V4 %s
; RUN: llc -mtriple=x86_64-linux -dwarf-version=3 -O0 -filetype=obj < %s \
; RUN:     | llvm-dwarfdump -debug-dump=info - | FileCheck --check-prefix=CHECK --check-prefix=CHECK-V3 %s

; Check that we emit DW_TAG_lexical_block and that it has the right encoding
; depending on the dwarf version.

; CHECK: DW_TAG_lexical_block
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_low_pc [DW_FORM_addr]
; CHECK-NOT: DW_TAG
; CHECK-V4: DW_AT_high_pc [DW_FORM_data4]
; CHECK-V3: DW_AT_high_pc [DW_FORM_addr]

; Test case produced from:
; void b() {
;   if (int i = 3)
;     return;
; }

; Function Attrs: nounwind uwtable
define void @_Z1bv() #0 {
entry:
  %i = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i32* %i, metadata !11, metadata !{!"0x102"}), !dbg !14
  store i32 3, i32* %i, align 4, !dbg !14
  %0 = load i32* %i, align 4, !dbg !14
  %tobool = icmp ne i32 %0, 0, !dbg !14
  br i1 %tobool, label %if.then, label %if.end, !dbg !14

if.then:                                          ; preds = %entry
  br label %if.end, !dbg !15

if.end:                                           ; preds = %if.then, %entry
  ret void, !dbg !16
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!8, !9}
!llvm.ident = !{!10}

!0 = !{!"0x11\004\00clang version 3.5.0 \000\00\000\00\001", !1, !2, !2, !3, !2, !2} ; [ DW_TAG_compile_unit ] [/tmp/dbginfo/lexical_block.cpp] [DW_LANG_C_plus_plus]
!1 = !{!"lexical_block.cpp", !"/tmp/dbginfo"}
!2 = !{}
!3 = !{!4}
!4 = !{!"0x2e\00b\00b\00_Z1bv\001\000\001\000\006\00256\000\001", !1, !5, !6, null, void ()* @_Z1bv, null, null, !2} ; [ DW_TAG_subprogram ] [line 1] [def] [b]
!5 = !{!"0x29", !1}          ; [ DW_TAG_file_type ] [/tmp/dbginfo/lexical_block.cpp]
!6 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !7, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = !{null}
!8 = !{i32 2, !"Dwarf Version", i32 4}
!9 = !{i32 1, !"Debug Info Version", i32 2}
!10 = !{!"clang version 3.5.0 "}
!11 = !{!"0x100\00i\002\000", !12, !5, !13} ; [ DW_TAG_auto_variable ] [i] [line 2]
!12 = !{!"0xb\002\000\000", !1, !4} ; [ DW_TAG_lexical_block ] [/tmp/dbginfo/lexical_block.cpp]
!13 = !{!"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!14 = !MDLocation(line: 2, scope: !12)
!15 = !MDLocation(line: 3, scope: !12)
!16 = !MDLocation(line: 4, scope: !4)
