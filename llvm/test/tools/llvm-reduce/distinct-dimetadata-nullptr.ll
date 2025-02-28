; Test checking that distinct metadata reduction pass handles null pointers properly.
; This test will lead to a crash if nullptrs inside distinct metadata are not handled correctly, in this case inside DICompileUnit

; RUN: llvm-reduce --delta-passes=distinct-metadata --aggressive-named-md-reduction --test FileCheck --test-arg %s --test-arg --input-file %s -o %t
; CHECK: {{.*}}distinct !DICompileUnit{{.*}}


!llvm.module.flags = !{!0, !1, !6}
!llvm.dbg.cu = !{!4}

!0 = !{i32 7, !"Dwarf Version", i32 4}
!1 = !{i32 2, !"Source Lang Literal", !2}
!2 = !{!3}
!3 = !{!4, i32 33}
!4 = distinct !DICompileUnit(language: DW_LANG_OpenCL, file: !5, producer: "foobar", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
!5 = !DIFile(filename: "main.cpp", directory: "foodir")
!6 = !{i32 2, !"Debug Info Version", i32 3}
