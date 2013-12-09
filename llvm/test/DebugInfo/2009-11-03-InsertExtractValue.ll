; RUN: llvm-as < %s | llvm-dis | FileCheck %s

!llvm.dbg.sp = !{!0}
!llvm.dbg.cu = !{!5}
!llvm.module.flags = !{!6}

!0 = metadata !{i32 786478, metadata !4, metadata !1, metadata !"bar", metadata !"bar", metadata !"_ZN3foo3barEv", i32 3, metadata !2, i1 false, i1 false, i32 0, i32 0, null, i32 258, i1 false, null, null, i32 0, metadata !1, i32 3} ; [ DW_TAG_subprogram ]
!1 = metadata !{i32 41, metadata !4} ; [ DW_TAG_file_type ]
!2 = metadata !{i32 21, metadata !4, metadata !1, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !3, i32 0, null} ; [ DW_TAG_subroutine_type ]
!3 = metadata !{null}
!4 = metadata !{metadata !"/foo", metadata !"bar.cpp"}
!5 = metadata !{i32 458769, metadata !4, i32 12, metadata !"", i1 true, metadata !"", i32 0, metadata !3, metadata !3, null, null, null, metadata !""}; [DW_TAG_compile_unit ]

define <{i32, i32}> @f1() {
; CHECK: !dbgx ![[NUMBER:[0-9]+]]
  %r = insertvalue <{ i32, i32 }> zeroinitializer, i32 4, 1, !dbgx !1
; CHECK: !dbgx ![[NUMBER]]
  %e = extractvalue <{ i32, i32 }> %r, 0, !dbgx !1
  ret <{ i32, i32 }> %r
}

; CHECK: [protected]
!6 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
