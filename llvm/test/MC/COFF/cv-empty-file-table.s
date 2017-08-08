# RUN: llvm-mc -filetype=obj -triple i686-pc-win32 < %s | llvm-readobj -codeview - | FileCheck %s
        .text
        .section        .debug$S,"dr"
        .p2align        2
        .long   4                       # Debug section magic
        .cv_filechecksums               # File index to string table offset subsection
        .cv_stringtable                 # String table

# CHECK: CodeViewDebugInfo [
# CHECK:   Section: .debug$S (4)
# CHECK:   Magic: 0x4
# CHECK-NOT: FileChecksum
# CHECK: ]
