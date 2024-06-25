# RUN: rm -rf %t && split-file %s %t && cd %t

#--- gnu-42-1.s
.section ".note.gnu.property", "a"
  .long 4           // Name length is always 4 ("GNU")
  .long end - begin // Data length
  .long 5           // Type: NT_GNU_PROPERTY_TYPE_0
  .asciz "GNU"      // Name
  .p2align 3
begin:
  # PAuth ABI property note
  .long 0xc0000001  // Type: GNU_PROPERTY_AARCH64_FEATURE_PAUTH
  .long 16          // Data size
  .quad 42          // PAuth ABI platform
  .quad 1           // PAuth ABI version
  .p2align 3        // Align to 8 byte for 64 bit
end:

# RUN: llvm-mc -filetype=obj -triple aarch64-linux-gnu gnu-42-1.s -o gnu-42-1.o
# RUN: llvm-readelf --notes gnu-42-1.o | \
# RUN:   FileCheck --check-prefix=ELF -DPLATFORM="0x2a (unknown)" -DVERSION=0x1 %s
# RUN: llvm-readobj --notes gnu-42-1.o | \
# RUN:   FileCheck --check-prefix=OBJ -DPLATFORM="0x2a (unknown)" -DVERSION=0x1 %s

# ELF: Displaying notes found in: .note.gnu.property
# ELF-NEXT:   Owner                 Data size	Description
# ELF-NEXT:   GNU                   0x00000018	NT_GNU_PROPERTY_TYPE_0 (property note)
# ELF-NEXT:   AArch64 PAuth ABI core info: platform [[PLATFORM]], version [[VERSION]]

# OBJ:      Notes [
# OBJ-NEXT:   NoteSection {
# OBJ-NEXT:     Name: .note.gnu.property
# OBJ-NEXT:     Offset: 0x40
# OBJ-NEXT:     Size: 0x28
# OBJ-NEXT:     Note {
# OBJ-NEXT:       Owner: GNU
# OBJ-NEXT:       Data size: 0x18
# OBJ-NEXT:       Type: NT_GNU_PROPERTY_TYPE_0 (property note)
# OBJ-NEXT:       Property [
# OBJ-NEXT:         AArch64 PAuth ABI core info: platform [[PLATFORM]], version [[VERSION]]
# OBJ-NEXT:       ]
# OBJ-NEXT:     }
# OBJ-NEXT:   }
# OBJ-NEXT: ]

#--- gnu-0-0.s
.section ".note.gnu.property", "a"
  .long 4           // Name length is always 4 ("GNU")
  .long end - begin // Data length
  .long 5           // Type: NT_GNU_PROPERTY_TYPE_0
  .asciz "GNU"      // Name
  .p2align 3
begin:
  # PAuth ABI property note
  .long 0xc0000001  // Type: GNU_PROPERTY_AARCH64_FEATURE_PAUTH
  .long 16          // Data size
  .quad 0           // PAuth ABI platform
  .quad 0           // PAuth ABI version
  .p2align 3        // Align to 8 byte for 64 bit
end:

# RUN: llvm-mc -filetype=obj -triple aarch64-linux-gnu gnu-0-0.s -o gnu-0-0.o
# RUN: llvm-readelf --notes gnu-0-0.o | \
# RUN:   FileCheck --check-prefix=ELF -DPLATFORM="0x0 (invalid)" -DVERSION=0x0 %s
# RUN: llvm-readobj --notes gnu-0-0.o | \
# RUN:   FileCheck --check-prefix=OBJ -DPLATFORM="0x0 (invalid)" -DVERSION=0x0 %s

#--- gnu-1-0.s
.section ".note.gnu.property", "a"
  .long 4           // Name length is always 4 ("GNU")
  .long end - begin // Data length
  .long 5           // Type: NT_GNU_PROPERTY_TYPE_0
  .asciz "GNU"      // Name
  .p2align 3
begin:
  # PAuth ABI property note
  .long 0xc0000001  // Type: GNU_PROPERTY_AARCH64_FEATURE_PAUTH
  .long 16          // Data size
  .quad 1           // PAuth ABI platform
  .quad 0           // PAuth ABI version
  .p2align 3        // Align to 8 byte for 64 bit
end:

# RUN: llvm-mc -filetype=obj -triple aarch64-linux-gnu gnu-1-0.s -o gnu-1-0.o
# RUN: llvm-readelf --notes gnu-1-0.o | \
# RUN:   FileCheck --check-prefix=ELF -DPLATFORM="0x1 (baremetal)" -DVERSION=0x0 %s
# RUN: llvm-readobj --notes gnu-1-0.o | \
# RUN:   FileCheck --check-prefix=OBJ -DPLATFORM="0x1 (baremetal)" -DVERSION=0x0 %s

#--- gnu-0x10000002-85.s
.section ".note.gnu.property", "a"
  .long 4           // Name length is always 4 ("GNU")
  .long end - begin // Data length
  .long 5           // Type: NT_GNU_PROPERTY_TYPE_0
  .asciz "GNU"      // Name
  .p2align 3
begin:
  # PAuth ABI property note
  .long 0xc0000001  // Type: GNU_PROPERTY_AARCH64_FEATURE_PAUTH
  .long 16          // Data size
  .quad 0x10000002  // PAuth ABI platform
  .quad 85          // PAuth ABI version
  .p2align 3        // Align to 8 byte for 64 bit
end:

# RUN: llvm-mc -filetype=obj -triple aarch64-linux-gnu gnu-0x10000002-85.s -o gnu-0x10000002-85.o
# RUN: llvm-readelf --notes gnu-0x10000002-85.o | \
# RUN:   FileCheck --check-prefix=ELF -DPLATFORM="0x10000002 (llvm_linux)" \
# RUN:   -DVERSION="0x55 (PointerAuthIntrinsics, !PointerAuthCalls, PointerAuthReturns, !PointerAuthAuthTraps, PointerAuthVTPtrAddressDiscrimination, !PointerAuthVTPtrTypeDiscrimination, PointerAuthInitFini)" %s
# RUN: llvm-readobj --notes gnu-0x10000002-85.o | \
# RUN:   FileCheck --check-prefix=OBJ -DPLATFORM="0x10000002 (llvm_linux)" \
# RUN:   -DVERSION="0x55 (PointerAuthIntrinsics, !PointerAuthCalls, PointerAuthReturns, !PointerAuthAuthTraps, PointerAuthVTPtrAddressDiscrimination, !PointerAuthVTPtrTypeDiscrimination, PointerAuthInitFini)" %s

#--- gnu-0x10000002-128.s
.section ".note.gnu.property", "a"
  .long 4           // Name length is always 4 ("GNU")
  .long end - begin // Data length
  .long 5           // Type: NT_GNU_PROPERTY_TYPE_0
  .asciz "GNU"      // Name
  .p2align 3
begin:
  # PAuth ABI property note
  .long 0xc0000001  // Type: GNU_PROPERTY_AARCH64_FEATURE_PAUTH
  .long 16          // Data size
  .quad 0x10000002  // PAuth ABI platform
  .quad 128         // PAuth ABI version
  .p2align 3        // Align to 8 byte for 64 bit
end:

# RUN: llvm-mc -filetype=obj -triple aarch64-linux-gnu gnu-0x10000002-128.s -o gnu-0x10000002-128.o
# RUN: llvm-readelf --notes gnu-0x10000002-128.o | \
# RUN:   FileCheck --check-prefix=ELF -DPLATFORM="0x10000002 (llvm_linux)" -DVERSION="0x80 (unknown)" %s
# RUN: llvm-readobj --notes gnu-0x10000002-128.o | \
# RUN:   FileCheck --check-prefix=OBJ -DPLATFORM="0x10000002 (llvm_linux)" -DVERSION="0x80 (unknown)" %s

#--- gnu-short.s
.section ".note.gnu.property", "a"
  .long 4           // Name length is always 4 ("GNU")
  .long end - begin // Data length
  .long 5           // Type: NT_GNU_PROPERTY_TYPE_0
  .asciz "GNU"      // Name
  .p2align 3
begin:
  # PAuth ABI property note
  .long 0xc0000001  // Type: GNU_PROPERTY_AARCH64_FEATURE_PAUTH
  .long 12          // Data size
  .quad 42          // PAuth ABI platform
  .word 1           // PAuth ABI version
  .p2align 3        // Align to 8 byte for 64 bit
end:

# RUN: llvm-mc -filetype=obj -triple aarch64-linux-gnu gnu-short.s -o gnu-short.o
# RUN: llvm-readelf --notes gnu-short.o | \
# RUN:   FileCheck --check-prefix=ELF-ERR -DSIZE=28 -DDATASIZE=18 \
# RUN:   -DERR="<corrupted size: expected 16, got 12>" %s
# RUN: llvm-readobj --notes gnu-short.o | \
# RUN:   FileCheck --check-prefix=OBJ-ERR -DSIZE=28 -DDATASIZE=18 \
# RUN:   -DERR="<corrupted size: expected 16, got 12>" %s

# ELF-ERR: Displaying notes found in: .note.gnu.property
# ELF-ERR-NEXT:   Owner                 Data size	Description
# ELF-ERR-NEXT:   GNU                   0x000000[[DATASIZE]]	NT_GNU_PROPERTY_TYPE_0 (property note)
# ELF-ERR-NEXT:   AArch64 PAuth ABI core info: [[ERR]]

# OBJ-ERR:      Notes [
# OBJ-ERR-NEXT:   NoteSection {
# OBJ-ERR-NEXT:     Name: .note.gnu.property
# OBJ-ERR-NEXT:     Offset: 0x40
# OBJ-ERR-NEXT:     Size: 0x[[SIZE]]
# OBJ-ERR-NEXT:     Note {
# OBJ-ERR-NEXT:       Owner: GNU
# OBJ-ERR-NEXT:       Data size: 0x[[DATASIZE]]
# OBJ-ERR-NEXT:       Type: NT_GNU_PROPERTY_TYPE_0 (property note)
# OBJ-ERR-NEXT:       Property [
# OBJ-ERR-NEXT:         AArch64 PAuth ABI core info: [[ERR]]
# OBJ-ERR-NEXT:       ]
# OBJ-ERR-NEXT:     }
# OBJ-ERR-NEXT:   }
# OBJ-ERR-NEXT: ]

#--- gnu-long.s
.section ".note.gnu.property", "a"
  .long 4           // Name length is always 4 ("GNU")
  .long end - begin // Data length
  .long 5           // Type: NT_GNU_PROPERTY_TYPE_0
  .asciz "GNU"      // Name
  .p2align 3
begin:
  # PAuth ABI property note
  .long 0xc0000001  // Type: GNU_PROPERTY_AARCH64_FEATURE_PAUTH
  .long 24          // Data size
  .quad 42          // PAuth ABI platform
  .quad 1           // PAuth ABI version
  .quad 0x0123456789ABCDEF
  .p2align 3        // Align to 8 byte for 64 bit
end:

# RUN: llvm-mc -filetype=obj -triple aarch64-linux-gnu gnu-long.s -o gnu-long.o
# RUN: llvm-readelf --notes gnu-long.o | \
# RUN:   FileCheck --check-prefix=ELF-ERR -DSIZE=30 -DDATASIZE=20 \
# RUN:   -DERR="<corrupted size: expected 16, got 24>" %s
# RUN: llvm-readobj --notes gnu-long.o | \
# RUN:   FileCheck --check-prefix=OBJ-ERR -DSIZE=30 -DDATASIZE=20 \
# RUN:   -DERR="<corrupted size: expected 16, got 24>" %s
