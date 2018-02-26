; RUN: bugpoint -load %llvmshlibdir/BugpointPasses%shlibext %s -output-prefix %t -bugpoint-crash-too-many-cus -silence-passes -disable-strip-debuginfo --opt-command opt > /dev/null
; RUN: llvm-dis %t-reduced-simplified.bc -o - | FileCheck %s
; RUN-DISABLE: bugpoint -disable-namedmd-remove -load %llvmshlibdir/BugpointPasses%shlibext %s -output-prefix %t -bugpoint-crash-too-many-cus -silence-passes > /dev/null
; RUN-DISABLE: llvm-dis %t-reduced-simplified.bc -o - | FileCheck %s
; REQUIRES: loadable_module

; CHECK: !llvm.dbg.cu = !{![[FIRST:[0-9]+]], ![[SECOND:[0-9]+]]}
; CHECK-DISABLE:      !llvm.dbg.cu = !{![[FIRST:[0-9]+]], ![[SECOND:[0-9]+]],
; CHECK-DISABLE-SAME: ![[THIRD:[0-9]+]], ![[FOURTH:[0-9]+]], ![[FIFTH:[0-9]+]]}
!llvm.dbg.cu = !{!0, !1, !2, !3, !4, !5}
; CHECK-NOT: !named
; CHECK-DISABLE: !named
!named = !{!0, !1, !2, !3, !4, !5}
; CHECK: !llvm.module.flags = !{![[DIVERSION:[0-9]+]]}
!llvm.module.flags = !{!6, !7}

; CHECK-DAG: ![[FIRST]] = distinct !DICompileUnit(language: DW_LANG_Julia,
; CHECK-DAG: ![[SECOND]] = distinct !DICompileUnit(language: DW_LANG_Julia,
; CHECK-DAG: ![[DIVERSION]] = !{i32 2, !"Debug Info Version", i32 3}
; CHECK-DAG: !DIFile(filename: "a", directory: "b")

; 4 nodes survive. Due to renumbering !4 should not exist
; CHECK-NOT: !4

!0 = distinct !DICompileUnit(language: DW_LANG_Julia,
                             file: !8)
!1 = distinct !DICompileUnit(language: DW_LANG_Julia,
                             file: !8)
!2 = distinct !DICompileUnit(language: DW_LANG_Julia,
                             file: !8)
!3 = distinct !DICompileUnit(language: DW_LANG_Julia,
                             file: !8)
!4 = distinct !DICompileUnit(language: DW_LANG_Julia,
                             file: !8)
!5 = distinct !DICompileUnit(language: DW_LANG_Julia,
                             file: !8)
!6 = !{i32 2, !"Dwarf Version", i32 2}
!7 = !{i32 2, !"Debug Info Version", i32 3}
!8 = !DIFile(filename: "a", directory: "b")
