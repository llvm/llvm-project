; RUN: llc < %s | FileCheck %s
target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

; Function Attrs: nounwind
define void @foo() #0 {
entry:
  call void @llvm.eh.unwind.init(), !dbg !9
  ret void, !dbg !10
}

; CHECK: @foo
; CHECK-NOT: .cfi_offset vrsave
; CHECK: blr

; Function Attrs: nounwind
declare void @llvm.eh.unwind.init() #0

attributes #0 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!8, !11}

!0 = !{!"0x11\0012\00clang version 3.4\000\00\000\00\000", !1, !2, !2, !3, !2, !2} ; [ DW_TAG_compile_unit ] [/tmp/unwind-dw2.c] [DW_LANG_C99]
!1 = !{!"/tmp/unwind-dw2.c", !"/tmp"}
!2 = !{}
!3 = !{!4}
!4 = !{!"0x2e\00foo\00foo\00\001\000\001\000\006\000\000\001", !1, !5, !6, null, void ()* @foo, null, null, !2} ; [ DW_TAG_subprogram ] [line 1] [def] [foo]
!5 = !{!"0x29", !1}          ; [ DW_TAG_file_type ] [/tmp/unwind-dw2.c]
!6 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !7, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = !{null}
!8 = !{i32 2, !"Dwarf Version", i32 3}
!9 = !MDLocation(line: 2, scope: !4)
!10 = !MDLocation(line: 3, scope: !4)
!11 = !{i32 1, !"Debug Info Version", i32 2}
