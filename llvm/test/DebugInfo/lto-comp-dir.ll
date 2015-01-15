; REQUIRES: object-emission

; RUN: %llc_dwarf < %s -filetype=obj | llvm-dwarfdump -debug-dump=line - | FileCheck %s
; RUN: %llc_dwarf < %s -filetype=asm | FileCheck --check-prefix=ASM %s

; If multiple line tables are emitted, one per CU, those line tables can
; unambiguously rely on the comp_dir of their owning CU and use directory '0'
; to refer to it.

; CHECK: .debug_line contents:
; CHECK-NEXT: Line table prologue:
; CHECK-NOT: include_directories
; CHECK: file_names[   1]   0 {{.*}} a.cpp
; CHECK-NOT: file_names

; CHECK: Line table prologue:
; CHECK-NOT: include_directories
; CHECK: file_names[   1]   0 {{.*}} b.cpp
; CHECK-NOT: file_names

; However, if a single line table is emitted and shared between CUs, the
; comp_dir is ambiguous and relying on it would lead to different path
; interpretations depending on which CU lead to the table - so ensure that
; full paths are always emitted in this case, never comp_dir relative.

; ASM: .file   1 "/tmp/dbginfo/a{{[/\\]+}}a.cpp"
; ASM: .file   2 "/tmp/dbginfo/b{{[/\\]+}}b.cpp"

; Generated from the following source compiled to bitcode from within their
; respective directories (with debug info) and linked together with llvm-link

; a/a.cpp
; void func() {
; }

; b/b.cpp
; void func();
; int main() {
;   func();
; }

; Function Attrs: nounwind uwtable
define void @_Z4funcv() #0 {
entry:
  ret void, !dbg !19
}

; Function Attrs: uwtable
define i32 @main() #1 {
entry:
  call void @_Z4funcv(), !dbg !20
  ret i32 0, !dbg !21
}

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0, !8}
!llvm.module.flags = !{!16, !17}
!llvm.ident = !{!18, !18}

!0 = !{!"0x11\004\00clang version 3.5.0 \000\00\000\00\001", !1, !2, !2, !3, !2, !2} ; [ DW_TAG_compile_unit ]
!1 = !{!"a.cpp", !"/tmp/dbginfo/a"}
!2 = !{}
!3 = !{!4}
!4 = !{!"0x2e\00func\00func\00_Z4funcv\001\000\001\000\006\00256\000\001", !1, !5, !6, null, void ()* @_Z4funcv, null, null, !2} ; [ DW_TAG_subprogram ] [line 1] [def] [func]
!5 = !{!"0x29", !1}          ; [ DW_TAG_file_type ] [/tmp/dbginfo/a/a.cpp]
!6 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !7, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = !{null}
!8 = !{!"0x11\004\00clang version 3.5.0 \000\00\000\00\001", !9, !2, !2, !10, !2, !2} ; [ DW_TAG_compile_unit ]
!9 = !{!"b.cpp", !"/tmp/dbginfo/b"}
!10 = !{!11}
!11 = !{!"0x2e\00main\00main\00\002\000\001\000\006\00256\000\002", !9, !12, !13, null, i32 ()* @main, null, null, !2} ; [ DW_TAG_subprogram ] [line 2] [def] [main]
!12 = !{!"0x29", !9}         ; [ DW_TAG_file_type ] [/tmp/dbginfo/b/b.cpp]
!13 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !14, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!14 = !{!15}
!15 = !{!"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!16 = !{i32 2, !"Dwarf Version", i32 4}
!17 = !{i32 1, !"Debug Info Version", i32 2}
!18 = !{!"clang version 3.5.0 "}
!19 = !MDLocation(line: 2, scope: !4)
!20 = !MDLocation(line: 3, scope: !11)
!21 = !MDLocation(line: 4, scope: !11)

