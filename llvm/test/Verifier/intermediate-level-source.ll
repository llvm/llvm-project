; RUN: not llvm-as < %s -o /dev/null 2>&1 | FileCheck %s
;;
;; Verify that !llvm.intermediate_level_source entries are structurally
;; well-formed: each entry must be a !{kind, file, source} tuple where kind and
;; source are non-null metadata strings and file is a DIFile. A malformed entry
;; would otherwise only assert deep in the NVPTX backend.

; CHECK: invalid source operand in llvm.intermediate_level_source metadata entry (expected metadata string)
; CHECK-NEXT: !{{[0-9]+}} = !{!"tile ir", !{{[0-9]+}}, null}
!0 = !{!"tile ir", !6, null}

; CHECK: invalid kind operand in llvm.intermediate_level_source metadata entry (expected metadata string)
; CHECK-NEXT: !{{[0-9]+}} = !{null, !{{[0-9]+}}, !"src"}
!1 = !{null, !6, !"src"}

; CHECK: invalid file operand in llvm.intermediate_level_source metadata entry (expected DIFile)
; CHECK-NEXT: !{{[0-9]+}} = !{!"tile ir", !"name2", !"src"}
!2 = !{!"tile ir", !"name2", !"src"}

; CHECK: invalid kind operand in llvm.intermediate_level_source metadata entry (expected metadata string)
; CHECK: invalid file operand in llvm.intermediate_level_source metadata entry (expected DIFile)
; CHECK: invalid source operand in llvm.intermediate_level_source metadata entry (expected metadata string)
!3 = !{null, null, null}

; CHECK: incorrect number of operands in llvm.intermediate_level_source metadata entry (expected !{kind, filename, source})
!4 = !{!"tile ir", !6}

; A well-formed entry must produce no diagnostic.
; CHECK-NOT: in llvm.intermediate_level_source
!5 = !{!"tile ir", !6, !"src"}

!6 = !DIFile(filename: "name4", directory: ".")

!llvm.intermediate_level_source = !{!0, !1, !2, !3, !4, !5}
