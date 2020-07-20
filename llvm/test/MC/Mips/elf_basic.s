// 32 bit big endian
// RUN: llvm-mc -filetype=obj -triple mips-unknown-linux %s -o - | llvm-readobj -h - | FileCheck -check-prefix=CHECK-BE32 %s
// 32 bit little endian
// RUN: llvm-mc -filetype=obj -triple mipsel-unknown-linux %s -o - | llvm-readobj -h - | FileCheck -check-prefix=CHECK-LE32 %s
// 64 bit big endian
// RUN: llvm-mc -filetype=obj -arch=mips64 -triple mips64-unknown-linux %s -o - | llvm-readobj -h - | FileCheck -check-prefix=CHECK-BE64 %s
// 64 bit little endian
// RUN: llvm-mc -filetype=obj -arch=mips64el -triple mips64el-unknown-linux %s -o - | llvm-readobj -h - | FileCheck -check-prefix=CHECK-LE64 %s

// Check that we produce 32 bit with each endian.

// CHECK-BE32: ElfHeader {
// CHECK-BE32:   Ident {
// CHECK-BE32:     Class: 32-bit
// CHECK-BE32:     DataEncoding: BigEndian
// CHECK-BE32:   }
// CHECK-BE32: }

// CHECK-LE32: ElfHeader {
// CHECK-LE32:   Ident {
// CHECK-LE32:     Class: 32-bit
// CHECK-LE32:     DataEncoding: LittleEndian
// CHECK-LE32:   }
// CHECK-LE32: }

// Check that we produce 64 bit with each endian.

// CHECK-BE64: ElfHeader {
// CHECK-BE64:   Ident {
// CHECK-BE64:     Class: 64-bit
// CHECK-BE64:     DataEncoding: BigEndian
// CHECK-BE64:   }
// CHECK-BE64: }

// CHECK-LE64: ElfHeader {
// CHECK-LE64:   Ident {
// CHECK-LE64:     Class: 64-bit
// CHECK-LE64:     DataEncoding: LittleEndian
// CHECK-LE64:     OS/ABI: SystemV
// CHECK-LE64:   }
// CHECK-LE64: }
