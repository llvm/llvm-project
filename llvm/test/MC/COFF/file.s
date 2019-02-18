// RUN: llvm-mc -triple i686-windows -filetype obj %s -o - | llvm-objdump -t - \
// RUN:   | FileCheck %s

// Round trip through .s output to exercise MCAsmStreamer.
// RUN: llvm-mc -triple i686-windows %s -o - \
// RUN:   | llvm-mc -triple i686-windows - -filetype=obj -o - | llvm-objdump -t - \
// RUN:   | FileCheck %s

// RUN: llvm-mc -triple i686-windows -filetype obj %s -o - \
// RUN:	  | llvm-readobj -symbols | FileCheck %s -check-prefix CHECK-SCN

	.file "null-padded.asm"
// CHECK: (nx 1) {{0x[0-9]+}} .file
// CHECK-NEXT: AUX null-padded.asm{{$}}

	.file "eighteen-chars.asm"

// CHECK: (nx 1) {{0x[0-9]+}} .file
// CHECK-NEXT: AUX eighteen-chars.asm{{$}}

	.file "multiple-auxiliary-entries.asm"

// CHECK: (nx 2) {{0x[0-9]+}} .file
// CHECK-NEXT: AUX multiple-auxiliary-entries.asm{{$}}

// CHECK-SCN: Symbols [
// CHECK-SCN:   Symbol {
// CHECK-SCN:     Name: .file
// CHECK-SCN:     Section: IMAGE_SYM_DEBUG (-2)
// CHECK-SCN:     StorageClass: File
// CHECK-SCN:     AuxFileRecord {
// CHECK-SCN:       FileName: null-padded.asm
// CHECK-SCN:     }
// CHECK-SCN:   }
// CHECK-SCN:   Symbol {
// CHECK-SCN:     Name: .file
// CHECK-SCN:     Section: IMAGE_SYM_DEBUG (-2)
// CHECK-SCN:     StorageClass: File
// CHECK-SCN:     AuxFileRecord {
// CHECK-SCN:       FileName: eighteen-chars.asm
// CHECK-SCN:     }
// CHECK-SCN:   }
// CHECK-SCN:   Symbol {
// CHECK-SCN:     Name: .file
// CHECK-SCN:     Section: IMAGE_SYM_DEBUG (-2)
// CHECK-SCN:     StorageClass: File
// CHECK-SCN:     AuxFileRecord {
// CHECK-SCN:       FileName: multiple-auxiliary-entries.asm
// CHECK-SCN:     }
// CHECK-SCN:   }
// CHECK-SCN: ]

