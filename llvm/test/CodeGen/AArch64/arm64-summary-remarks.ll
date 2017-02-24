; RUN: llc < %s -mtriple=arm64-apple-ios7.0 -pass-remarks-analysis=asm-printer 2>&1 | FileCheck %s

; CHECK: arm64-summary-remarks.ll:5:0: 1 instructions in function

define void @empty_func() nounwind ssp !dbg !3 {
  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1)
!1 = !DIFile(filename: "arm64-summary-remarks.ll", directory: "")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = distinct !DISubprogram(name: "empty_func", scope: !1, file: !1, line: 5, scopeLine: 5, unit: !0)
