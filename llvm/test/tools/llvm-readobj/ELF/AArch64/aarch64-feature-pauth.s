# RUN: rm -rf %t && split-file %s %t && cd %t

# RUN: llvm-mc -filetype=obj -triple=aarch64-linux-gnu abi-tag.s       -o tag.o
# RUN: llvm-mc -filetype=obj -triple=aarch64-linux-gnu abi-tag-short.s -o tag-short.o
# RUN: llvm-mc -filetype=obj -triple=aarch64-linux-gnu abi-tag-long.s  -o tag-long.o

# RUN: llvm-readelf --notes tag.o       | FileCheck --check-prefix NORMAL %s
# RUN: llvm-readelf --notes tag-short.o | FileCheck --check-prefix SHORT  %s
# RUN: llvm-readelf --notes tag-long.o  | FileCheck --check-prefix LONG   %s

# NORMAL: AArch64 PAuth ABI tag: platform 0x2a, version 0x1
# SHORT:  AArch64 PAuth ABI tag: <corrupted size: expected at least 16, got 12>
# LONG:   AArch64 PAuth ABI tag: platform 0x2a, version 0x1, additional info 0xEFCDAB8967452301

# RUN: llvm-readobj --notes tag.o       | FileCheck --check-prefix LLVM-NORMAL %s
# RUN: llvm-readobj --notes tag-short.o | FileCheck --check-prefix LLVM-SHORT %s
# RUN: llvm-readobj --notes tag-long.o  | FileCheck --check-prefix LLVM-LONG %s

# LLVM-SHORT:      Notes [
# LLVM-SHORT-NEXT:   NoteSection {
# LLVM-SHORT-NEXT:     Name: .note.AARCH64-PAUTH-ABI-tag
# LLVM-SHORT-NEXT:     Offset: 0x40
# LLVM-SHORT-NEXT:     Size: 0x1C
# LLVM-SHORT-NEXT:     Note {
# LLVM-SHORT-NEXT:       Owner: ARM
# LLVM-SHORT-NEXT:       Data size: 0xC
# LLVM-SHORT-NEXT:       Type: NT_ARM_TYPE_PAUTH_ABI_TAG
# LLVM-SHORT-NEXT:       Description data (
# LLVM-SHORT-NEXT:         0000: 2A000000 00000000 01000000
# LLVM-SHORT-NEXT:       )
# LLVM-SHORT-NEXT:     }
# LLVM-SHORT-NEXT:   }
# LLVM-SHORT-NEXT: ]

# LLVM-NORMAL:      Notes [
# LLVM-NORMAL-NEXT:   NoteSection {
# LLVM-NORMAL-NEXT:     Name: .note.AARCH64-PAUTH-ABI-tag
# LLVM-NORMAL-NEXT:     Offset: 0x40
# LLVM-NORMAL-NEXT:     Size: 0x20
# LLVM-NORMAL-NEXT:     Note {
# LLVM-NORMAL-NEXT:       Owner: ARM
# LLVM-NORMAL-NEXT:       Data size: 0x10
# LLVM-NORMAL-NEXT:       Type: NT_ARM_TYPE_PAUTH_ABI_TAG
# LLVM-NORMAL-NEXT:       Platform: 42
# LLVM-NORMAL-NEXT:       Version: 1
# LLVM-NORMAL-NEXT:     }
# LLVM-NORMAL-NEXT:   }
# LLVM-NORMAL-NEXT: ]

# LLVM-LONG:      Notes [
# LLVM-LONG-NEXT:   NoteSection {
# LLVM-LONG-NEXT:     Name: .note.AARCH64-PAUTH-ABI-tag
# LLVM-LONG-NEXT:     Offset: 0x40
# LLVM-LONG-NEXT:     Size: 0x28
# LLVM-LONG-NEXT:     Note {
# LLVM-LONG-NEXT:       Owner: ARM
# LLVM-LONG-NEXT:       Data size: 0x18
# LLVM-LONG-NEXT:       Type: NT_ARM_TYPE_PAUTH_ABI_TAG
# LLVM-LONG-NEXT:       Platform: 42
# LLVM-LONG-NEXT:       Version: 1
# LLVM-LONG-NEXT:       Additional info: EFCDAB8967452301
# LLVM-LONG-NEXT:     }
# LLVM-LONG-NEXT:   }
# LLVM-LONG-NEXT: ]

#--- abi-tag.s

.section ".note.AARCH64-PAUTH-ABI-tag", "a"
.long 4
.long 16
.long 1
.asciz "ARM"

.quad 42         // platform
.quad 1          // version

#--- abi-tag-short.s

.section ".note.AARCH64-PAUTH-ABI-tag", "a"
.long 4
.long 12
.long 1
.asciz "ARM"

.quad 42
.word 1

#--- abi-tag-long.s

.section ".note.AARCH64-PAUTH-ABI-tag", "a"
.long 4
.long 24
.long 1
.asciz "ARM"

.quad 42         // platform
.quad 1          // version
.quad 0x0123456789ABCDEF // extra data
