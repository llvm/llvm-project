; RUN: llc -mtriple powerpc-ibm-aix-xcoff < %s | \
; RUN: FileCheck --check-prefix=ASM %s
; RUN: llc -mtriple powerpc64-ibm-aix-xcoff < %s | \
; RUN: FileCheck --check-prefix=ASM %s

; RUN: llc -mtriple powerpc-ibm-aix-xcoff -filetype=obj  -o %t.o < %s
; RUN: llvm-objdump --full-contents --section=.info %t.o | \
; RUN: FileCheck --check-prefix=OBJ %s
; RUN: llc -mtriple powerpc64-ibm-aix-xcoff -filetype=obj  -o %t.o < %s
; RUN: llvm-objdump --full-contents --section=.info %t.o | \
; RUN: FileCheck --check-prefix=OBJ %s

; Verify that llvm.commandline metadata is emitted to .info sections and that the
; metadata is padded if necessary.

; ASM: .info ".GCC.command.line", 0x0000003a,
; ASM: .info , 0x40282329, 0x6f707420, 0x636c616e, 0x67202d63, 0x6f6d6d61, 0x6e64202d
; ASM: .info , 0x6c696e65, 0x0a004028, 0x23296f70, 0x7420736f, 0x6d657468, 0x696e6720
; ASM: .info , 0x656c7365, 0x20313233, 0x0a000000

; OBJ: Contents of section .info:
; OBJ: 0000 0000003a 40282329 6f707420 636c616e  ...:@(#)opt clan
; OBJ: 0010 67202d63 6f6d6d61 6e64202d 6c696e65  g -command -line
; OBJ: 0020 0a004028 23296f70 7420736f 6d657468  ..@(#)opt someth
; OBJ: 0030 696e6720 656c7365 20313233 0a000000  ing else 123....

!llvm.commandline = !{!0, !1}
!0 = !{!"clang -command -line"}
!1 = !{!"something else 123"}
