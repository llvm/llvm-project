; RUN: llc -mtriple powerpc-ibm-aix-xcoff < %s | \
; RUN: FileCheck --check-prefix=ASM %s
; RUN: llc -mtriple powerpc64-ibm-aix-xcoff < %s | \
; RUN: FileCheck --check-prefix=ASM %s

; RUN: not --crash llc -mtriple powerpc-ibm-aix-xcoff -filetype=obj  < %s 2>&1 | \
; RUN: FileCheck --check-prefix=OBJ %s
; RUN: not --crash llc -mtriple powerpc64-ibm-aix-xcoff -filetype=obj  < %s 2>&1 | \
; RUN: FileCheck --check-prefix=OBJ %s

; Verify that llvm.commandline metadata is emitted to .info sections and that the
; metadata is padded if necessary.

; OBJ: LLVM ERROR: emitXCOFFCInfoSym is not implemented yet on object generation path

; ASM: .info ".GCC.command.line", 0x0000003a,
; ASM: .info , 0x40282329, 0x6f707420, 0x636c616e, 0x67202d63, 0x6f6d6d61, 0x6e64202d
; ASM: .info , 0x6c696e65, 0x0a004028, 0x23296f70, 0x7420736f, 0x6d657468, 0x696e6720
; ASM: .info , 0x656c7365, 0x20313233, 0x0a000000

!llvm.commandline = !{!0, !1}
!0 = !{!"clang -command -line"}
!1 = !{!"something else 123"}
