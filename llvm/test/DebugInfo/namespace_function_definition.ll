; REQUIRES: object-emission

; RUN: %llc_dwarf -O0 -filetype=obj < %s | llvm-dwarfdump -debug-dump=info - | FileCheck %s

; Generated from clang with the following source:
; namespace ns {
; void func() {
; }
; }

; CHECK: DW_TAG_namespace
; CHECK-NEXT: DW_AT_name {{.*}} "ns"
; CHECK: DW_TAG_subprogram
; CHECK-NOT: DW_TAG
; CHECK:   DW_AT_low_pc
; CHECK-NOT: DW_TAG
; CHECK:   DW_AT_MIPS_linkage_name {{.*}} "_ZN2ns4funcEv"
; CHECK: NULL
; CHECK: NULL

; Function Attrs: nounwind uwtable
define void @_ZN2ns4funcEv() #0 {
entry:
  ret void, !dbg !11
}

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!8, !9}
!llvm.ident = !{!10}

!0 = !{!"0x11\004\00clang version 3.5.0 \000\00\000\00\001", !1, !2, !2, !3, !2, !2} ; [ DW_TAG_compile_unit ] [/tmp/dbginfo/namespace_function_definition.cpp] [DW_LANG_C_plus_plus]
!1 = !{!"namespace_function_definition.cpp", !"/tmp/dbginfo"}
!2 = !{}
!3 = !{!4}
!4 = !{!"0x2e\00func\00func\00_ZN2ns4funcEv\002\000\001\000\006\00256\000\002", !1, !5, !6, null, void ()* @_ZN2ns4funcEv, null, null, !2} ; [ DW_TAG_subprogram ] [line 2] [def] [func]
!5 = !{!"0x39\00ns\001", !1, null} ; [ DW_TAG_namespace ] [ns] [line 1]
!6 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !7, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = !{null}
!8 = !{i32 2, !"Dwarf Version", i32 4}
!9 = !{i32 1, !"Debug Info Version", i32 2}
!10 = !{!"clang version 3.5.0 "}
!11 = !MDLocation(line: 3, scope: !4)
