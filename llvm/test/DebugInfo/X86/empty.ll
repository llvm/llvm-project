; RUN: llc -mtriple i686-pc-cygwin < %s -filetype=obj | llvm-dwarfdump - | FileCheck %s
; RUN: llc -mtriple i686-pc-cygwin -split-dwarf=Enable < %s -filetype=obj | llvm-dwarfdump - | FileCheck --check-prefix=FISSION %s

; Expect no line table entry since there are no functions and file references in this compile unit
; CHECK: .debug_line contents:
; CHECK: Line table prologue:
; CHECK: total_length: 0x00000019
; CHECK-NOT: file_names[

; CHECK: .debug_pubnames contents:
; CHECK-NOT: Offset

; CHECK: .debug_pubtypes contents:
; CHECK-NOT: Offset

; Don't emit DW_AT_addr_base when there are no addresses.
; FISSION-NOT: DW_AT_GNU_addr_base [DW_FORM_sec_offset]

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 3.1 (trunk 143523)", isOptimized: true, emissionKind: FullDebug, file: !4, enums: !2, retainedTypes: !2, subprograms: !2, globals: !2)
!2 = !{}
!3 = !DIFile(filename: "empty.c", directory: "/home/nlewycky")
!4 = !DIFile(filename: "empty.c", directory: "/home/nlewycky")
!5 = !{i32 1, !"Debug Info Version", i32 3}
