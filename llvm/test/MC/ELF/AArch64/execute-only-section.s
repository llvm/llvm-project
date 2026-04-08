// RUN: llvm-mc -filetype=obj -triple aarch64-unknown-linux-gnu %s -o - \
// RUN: | llvm-readobj -S --symbols - | FileCheck %s --check-prefix=READOBJ
// RUN: llvm-mc -filetype=obj -triple aarch64-unknown-linux-gnu %s -o - \
// RUN: | llvm-readelf -S --symbols - | FileCheck %s --check-prefix=READELF

        .section        .text,"axy",@progbits,unique,0
        .globl  foo
        .p2align  2
        .type   foo,@function
foo:
        .cfi_startproc
        ret
.Lfunc_end0:
        .size   foo, .Lfunc_end0-foo
        .cfi_endproc

// READOBJ:      Section {
// READOBJ:        Name: .text
// READOBJ-NEXT:   Type: SHT_PROGBITS (0x1)
// READOBJ-NEXT:   Flags [ (0x20000006)
// READOBJ-NEXT:     SHF_AARCH64_PURECODE (0x20000000)
// READOBJ-NEXT:     SHF_ALLOC (0x2)
// READOBJ-NEXT:     SHF_EXECINSTR (0x4)
// READOBJ-NEXT:   ]
// READOBJ-NEXT:   Address:
// READOBJ-NEXT:   Offset:
// READOBJ-NEXT:   Size: 0
// READOBJ:      }

// READOBJ:      Section {
// READOBJ:        Name: .text
// READOBJ-NEXT:   Type: SHT_PROGBITS (0x1)
// READOBJ-NEXT:   Flags [ (0x20000006)
// READOBJ-NEXT:     SHF_AARCH64_PURECODE (0x20000000)
// READOBJ-NEXT:     SHF_ALLOC (0x2)
// READOBJ-NEXT:     SHF_EXECINSTR (0x4)
// READOBJ-NEXT:   ]
// READOBJ-NEXT:   Address:
// READOBJ-NEXT:   Offset:
// READOBJ-NEXT:   Size: 4
// READOBJ:      }

// READOBJ:      Symbol {
// READOBJ:        Name: foo
// READOBJ-NEXT:   Value:
// READOBJ-NEXT:   Size: 4
// READOBJ-NEXT:   Binding: Global
// READOBJ-NEXT:   Type: Function
// READOBJ-NEXT:   Other:
// READOBJ-NEXT:   Section: .text
// READOBJ:      }

// READELF: Section Headers:
// READELF: .text     PROGBITS {{[0-9a-f]+}} {{[0-9a-f]+}} 000000 {{[0-9a-f]+}} AXy  {{[0-9]+}} {{[0-9]+}} {{[0-9]+}}
// READELF: .text     PROGBITS {{[0-9a-f]+}} {{[0-9a-f]+}} 000004 {{[0-9a-f]+}} AXy  {{[0-9]+}} {{[0-9]+}} {{[0-9]+}}
