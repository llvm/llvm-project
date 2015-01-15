; REQUIRES: object-emission

; RUN: %llc_dwarf < %s -filetype=obj | llvm-dwarfdump -debug-dump=info - | FileCheck %s

; Test that a nodebug function (a function not appearing in the debug info IR
; metadata subprogram list) with DebugLocs on its IR doesn't cause crashes/does
; the right thing.

; Build with clang from the following:
; extern int i;
; inline __attribute__((always_inline)) void f1() {
;   i = 3;
; }
;
; __attribute__((nodebug)) void f2() {
;   f1();
; }

; Check that there's only one DW_TAG_subprogram, nothing for the 'f2' function.
; CHECK: DW_TAG_subprogram
; CHECK-NOT: DW_TAG
; CHECK:   DW_AT_name {{.*}} "f1"
; CHECK-NOT: DW_TAG_subprogram

@i = external global i32

; Function Attrs: uwtable
define void @_Z2f2v() #0 {
entry:
  store i32 3, i32* @i, align 4, !dbg !11
  ret void
}

attributes #0 = { uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!8, !9}
!llvm.ident = !{!10}

!0 = !{!"0x11\004\00clang version 3.5.0 \000\00\000\00\001", !1, !2, !2, !3, !2, !2} ; [ DW_TAG_compile_unit ] [/tmp/dbginfo/nodebug.cpp] [DW_LANG_C_plus_plus]
!1 = !{!"nodebug.cpp", !"/tmp/dbginfo"}
!2 = !{}
!3 = !{!4}
!4 = !{!"0x2e\00f1\00f1\00_Z2f1v\002\000\001\000\006\00256\000\002", !1, !5, !6, null, null, null, null, !2} ; [ DW_TAG_subprogram ] [line 2] [def] [f1]
!5 = !{!"0x29", !1}          ; [ DW_TAG_file_type ] [/tmp/dbginfo/nodebug.cpp]
!6 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !7, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = !{null}
!8 = !{i32 2, !"Dwarf Version", i32 4}
!9 = !{i32 2, !"Debug Info Version", i32 2}
!10 = !{!"clang version 3.5.0 "}
!11 = !MDLocation(line: 3, scope: !4)
