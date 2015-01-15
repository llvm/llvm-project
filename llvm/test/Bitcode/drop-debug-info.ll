; RUN: llvm-as < %s -o %t.bc 2>&1 >/dev/null | FileCheck -check-prefix=WARN %s
; RUN: llvm-dis < %t.bc | FileCheck %s
; RUN: verify-uselistorder < %t.bc

define i32 @main() {
entry:
  %retval = alloca i32, align 4
  store i32 0, i32* %retval
  ret i32 0, !dbg !12
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!9}

!0 = !{!"0x11\0012\00clang version 3.5 (trunk 195495) (llvm/trunk 195495:195504M)\000\00\000\00\000", !1, !2, !2, !3, !2, !2} ; [ DW_TAG_compile_unit ] [/Users/manmanren/llvm_gmail/release/../llvm/tools/clang/test/CodeGen/debug-info-version.c] [DW_LANG_C99]
!1 = !{!"../llvm/tools/clang/test/CodeGen/debug-info-version.c", !"/Users/manmanren/llvm_gmail/release"}
!2 = !{i32 0}
!3 = !{!4}
!4 = !{!"0x2e\00main\00main\00\003\000\001\000\006\00256\000\003", !1, !5, !6, null, i32 ()* @main, null, null, !2} ; [ DW_TAG_subprogram ] [line 3] [def] [main]
!5 = !{!"0x29", !1}          ; [ DW_TAG_file_type ] [/Users/manmanren/llvm_gmail/release/../llvm/tools/clang/test/CodeGen/debug-info-version.c]
!6 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !7, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = !{!8}
!8 = !{!"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!9 = !{i32 2, !"Dwarf Version", i32 2}
!12 = !MDLocation(line: 4, scope: !4)

; WARN: warning: ignoring debug info with an invalid version (0)
; CHECK-NOT: !dbg
; CHECK-NOT: !llvm.dbg.cu
