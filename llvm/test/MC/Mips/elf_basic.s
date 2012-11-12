// 32 bit big endian
// RUN: llvm-mc -filetype=obj -triple mips-unknown-linux %s -o - | elf-dump --dump-section-data  | FileCheck -check-prefix=CHECK-BE32 %s
// 32 bit little endian
// RUN: llvm-mc -filetype=obj -triple mipsel-unknown-linux %s -o - | elf-dump --dump-section-data  | FileCheck -check-prefix=CHECK-LE32 %s
// 64 bit big endian
// RUN: llvm-mc -filetype=obj -arch=mips64 -triple mips64-unknown-linux %s -o - | elf-dump --dump-section-data | FileCheck -check-prefix=CHECK-BE64 %s
// 64 bit little endian
// RUN: llvm-mc -filetype=obj -arch=mips64el -triple mips64el-unknown-linux %s -o - | elf-dump --dump-section-data | FileCheck -check-prefix=CHECK-LE64 %s

// Check that we produce 32 bit with each endian.

// This is 32 bit.
// CHECK-BE32: ('e_indent[EI_CLASS]', 0x01)
// This is big endian.
// CHECK-BE32: ('e_indent[EI_DATA]', 0x02)

// This is 32 bit.
// CHECK-LE32: ('e_indent[EI_CLASS]', 0x01)
// This is little endian.
// CHECK-LE32: ('e_indent[EI_DATA]', 0x01)

// Check that we produce 64 bit with each endian.

// This is 64 bit.
// CHECK-BE64: ('e_indent[EI_CLASS]', 0x02)
// This is big endian.
// CHECK-BE64: ('e_indent[EI_DATA]', 0x02)

// This is 64 bit.
// CHECK-LE64: ('e_indent[EI_CLASS]', 0x02)
// This is little endian.
// CHECK-LE64: ('e_indent[EI_DATA]', 0x01)

// Check that we are setting EI_OSABI to ELFOSABI_LINUX.
// CHECK-LE64: ('e_indent[EI_OSABI]', 0x03)
