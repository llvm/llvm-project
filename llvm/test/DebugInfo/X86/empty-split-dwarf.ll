; UNSUPPORTED: target={{.*}}-aix{{.*}}
; RUN: %llc_dwarf -split-dwarf-file=foo.dwo < %s -filetype=obj --mtriple=x86_64-unknown-linux-elf | llvm-dwarfdump -v - | FileCheck --check-prefix=FISSION %s

; Don't emit DW_AT_addr_base when there are no addresses.
; Also don't emit a split line table when there are no type units.
; FISSION-NOT: DW_AT_GNU_addr_base [DW_FORM_sec_offset]
; FISSION-NOT: .debug_line.dwo contents:

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 3.1 (trunk 143523)", isOptimized: true, emissionKind: FullDebug, file: !4, enums: !2, retainedTypes: !6, globals: !2)
!2 = !{}
!3 = !DIFile(filename: "empty.c", directory: "/home/nlewycky")
!4 = !DIFile(filename: "empty.c", directory: "/home/nlewycky")
!5 = !{i32 1, !"Debug Info Version", i32 3}
!6 = !{!7}
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
