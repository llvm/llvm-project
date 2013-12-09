; RUN: llvm-as < %s | llvm-dis | FileCheck %s

define i32 @main() {
entry:
  %retval = alloca i32, align 4
  store i32 0, i32* %retval
  ret i32 0, !dbg !12
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!9}

!0 = metadata !{i32 786449, metadata !1, i32 12, metadata !"clang version 3.5 (trunk 195495) (llvm/trunk 195495:195504M)", i1 false, metadata !"", i32 0, metadata !2, metadata !2, metadata !3, metadata !2, metadata !2, metadata !""} ; [ DW_TAG_compile_unit ] [/Users/manmanren/llvm_gmail/release/../llvm/tools/clang/test/CodeGen/debug-info-version.c] [DW_LANG_C99]
!1 = metadata !{metadata !"../llvm/tools/clang/test/CodeGen/debug-info-version.c", metadata !"/Users/manmanren/llvm_gmail/release"}
!2 = metadata !{i32 0}
!3 = metadata !{metadata !4}
!4 = metadata !{i32 786478, metadata !1, metadata !5, metadata !"main", metadata !"main", metadata !"", i32 3, metadata !6, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, i32 ()* @main, null, null, metadata !2, i32 3} ; [ DW_TAG_subprogram ] [line 3] [def] [main]
!5 = metadata !{i32 786473, metadata !1}          ; [ DW_TAG_file_type ] [/Users/manmanren/llvm_gmail/release/../llvm/tools/clang/test/CodeGen/debug-info-version.c]
!6 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !7, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = metadata !{metadata !8}
!8 = metadata !{i32 786468, null, null, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!9 = metadata !{i32 2, metadata !"Dwarf Version", i32 2}
!12 = metadata !{i32 4, i32 0, metadata !4, null}

; CHECK-NOT: !dbg
; CHECK-NOT: !llvm.dbg.cu
