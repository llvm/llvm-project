; RUN: bugpoint -load %llvmshlibdir/BugpointPasses%shlibext %s -output-prefix %t -bugpoint-crash-too-many-cus -silence-passes 2>&1 | FileCheck %s
; REQUIRES: plugins
; CHECK: DICompileUnit not listed in llvm.dbg.cu

; When bugpoint hacks at this testcase it will at one point create illegal IR
; that won't even pass the Verifier. A bugpoint *driver* built with assertions
; should not assert on it, but reject the malformed intermediate step.
define void @f() !dbg !9 { ret void }
!llvm.dbg.cu = !{!0, !1, !2, !3, !4, !5}
!0 = distinct !DICompileUnit(language: 12, file: !6)
!1 = distinct !DICompileUnit(language: 12, file: !6)
!2 = distinct !DICompileUnit(language: 12, file: !6)
!3 = distinct !DICompileUnit(language: 12, file: !6)
!4 = distinct !DICompileUnit(language: 12, file: !6)
!5 = distinct !DICompileUnit(language: 12, file: !6)
!6 = !DIFile(filename: "path/to/file", directory: "/path/to/dir")

!llvm.module.flags = !{!7, !8}
!7 = !{i32 2, !"Dwarf Version", i32 4}
!8 = !{i32 2, !"Debug Info Version", i32 3}

!9 = distinct !DISubprogram(unit: !0)
