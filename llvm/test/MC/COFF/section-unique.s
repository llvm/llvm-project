// RUN: llvm-mc -triple x86_64-pc-windows-msvc %s -o - | FileCheck %s
// RUN: llvm-mc -triple x86_64-pc-windows-msvc %s -filetype=obj -o - | llvm-readobj --symbols - | FileCheck %s --check-prefix=OBJ
  .section        .text,"xr",unique,0
      nop
  .section        .text,"xr",one_only,main,unique,4294967293
      nop
  .section        .text,"xr",discard,"",unique,4294967294
      nop

// CHECK: .section        .text,"xr",unique,0

// CHECK: .section        .text,"xr",one_only,main,unique,4294967293

// CHECK:       .section        .text,"xr",unique,4294967294
// CHECK-NEXT:  .linkonce       discard

// OBJ-COUNT-4: Symbol {
// OBJ-NEXT:      Name: .text
// OBJ-NEXT:      Value: 0
// OBJ-NEXT:      Section: .text (4)
// OBJ-NEXT:      BaseType: Null (0x0)
// OBJ-NEXT:      ComplexType: Null (0x0)
// OBJ-NEXT:      StorageClass: Static (0x3)
// OBJ-NEXT:      AuxSymbolCount: 1
// OBJ-NEXT:      AuxSectionDef {
// OBJ-NEXT:        Length: 1
// OBJ-NEXT:        RelocationCount: 0
// OBJ-NEXT:        LineNumberCount: 0
// OBJ-NEXT:        Checksum: 0xF00F9344
// OBJ-NEXT:        Number: 4
// OBJ-NEXT:        Selection: 0x0
// OBJ-NEXT:      }
// OBJ-NEXT:    }
// OBJ-NEXT:    Symbol {
// OBJ-NEXT:      Name: .text
// OBJ-NEXT:      Value: 0
// OBJ-NEXT:      Section: .text (5)
// OBJ-NEXT:      BaseType: Null (0x0)
// OBJ-NEXT:      ComplexType: Null (0x0)
// OBJ-NEXT:      StorageClass: Static (0x3)
// OBJ-NEXT:      AuxSymbolCount: 1
// OBJ-NEXT:      AuxSectionDef {
// OBJ-NEXT:        Length: 1
// OBJ-NEXT:        RelocationCount: 0
// OBJ-NEXT:        LineNumberCount: 0
// OBJ-NEXT:        Checksum: 0xF00F9344
// OBJ-NEXT:        Number: 5
// OBJ-NEXT:        Selection: NoDuplicates (0x1)
// OBJ-NEXT:      }
// OBJ-NEXT:    }
// OBJ-COUNT-2: Symbol {
// OBJ-NEXT:      Name: .text
// OBJ-NEXT:      Value: 0
// OBJ-NEXT:      Section: .text (6)
// OBJ-NEXT:      BaseType: Null (0x0)
// OBJ-NEXT:      ComplexType: Null (0x0)
// OBJ-NEXT:      StorageClass: Static (0x3)
// OBJ-NEXT:      AuxSymbolCount: 1
// OBJ-NEXT:      AuxSectionDef {
// OBJ-NEXT:        Length: 1
// OBJ-NEXT:        RelocationCount: 0
// OBJ-NEXT:        LineNumberCount: 0
// OBJ-NEXT:        Checksum: 0xF00F9344
// OBJ-NEXT:        Number: 6
// OBJ-NEXT:        Selection: Any (0x2)
// OBJ-NEXT:      }
// OBJ-NEXT:    }
