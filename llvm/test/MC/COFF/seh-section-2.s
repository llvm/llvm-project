# RUN: llvm-mc -filetype=obj -triple x86_64-pc-win32 %s | llvm-readobj --symbols - | FileCheck %s

# This assembly should make an object with two .text sections, two .xdata
# sections, and two .pdata sections.

        .def     f;
        .scl    2;
        .type   32;
        .endef
        .section        .text,"xr",discard,f
        .globl  f
        .p2align        4, 0x90
f:                                      # @f
.Ltmp0:
.seh_proc f
# %bb.0:
        subq    $40, %rsp
.Ltmp1:
        .seh_stackalloc 40
.Ltmp2:
        .seh_endprologue
        callq   g
        nop
        .seh_startepilogue
        addq    $40, %rsp
        .seh_endepilogue
        retq
        .seh_handlerdata
        .section        .text,"xr",discard,f
.Ltmp3:
        .seh_endproc

        .def     g;
        .scl    3;
        .type   32;
        .endef
        .section        .text,"xr",associative,f
        .p2align        4, 0x90
g:                                      # @g
.Ltmp4:
.seh_proc g
# %bb.0:
.Ltmp5:
        .seh_endprologue
        retq
        .seh_handlerdata
        .section        .text,"xr",associative,f
.Ltmp6:
        .seh_endproc


# CHECK: Symbols [
# CHECK:   Symbol {
# CHECK:     Name: .text
# CHECK:     Section: .text (4)
# CHECK:     AuxSymbolCount: 1
# CHECK:     AuxSectionDef {
# CHECK:       Length: 15
# CHECK:       RelocationCount: 1
# CHECK:       LineNumberCount: 0
# CHECK:       Checksum: 0xE17CBB7
# CHECK:       Number: 4
# CHECK:       Selection: Any (0x2)
# CHECK:     }
# CHECK:   }
# CHECK:   Symbol {
# CHECK:     Name: .xdata
# CHECK:     Value: 0
# CHECK:     Section: .xdata (5)
# CHECK:     BaseType: Null (0x0)
# CHECK:     ComplexType: Null (0x0)
# CHECK:     StorageClass: Static (0x3)
# CHECK:     AuxSymbolCount: 1
# CHECK:     AuxSectionDef {
# CHECK:       Length: 8
# CHECK:       RelocationCount: 0
# CHECK:       LineNumberCount: 0
# CHECK:       Checksum: 0xFC539D1
# CHECK:       Number: 4
# CHECK:       Selection: Associative (0x5)
# CHECK:       AssocSection: .text (4)
# CHECK:     }
# CHECK:   }
# CHECK:   Symbol {
# CHECK:     Name: .text
# CHECK:     Value: 0
# CHECK:     Section: .text (6)
# CHECK:     BaseType: Null (0x0)
# CHECK:     ComplexType: Null (0x0)
# CHECK:     StorageClass: Static (0x3)
# CHECK:     AuxSymbolCount: 1
# CHECK:     AuxSectionDef {
# CHECK:       Length: 1
# CHECK:       RelocationCount: 0
# CHECK:       LineNumberCount: 0
# CHECK:       Checksum: 0x26D930A
# CHECK:       Number: 4
# CHECK:       Selection: Associative (0x5)
# CHECK:       AssocSection: .text (4)
# CHECK:     }
# CHECK:   }
# CHECK:   Symbol {
# CHECK:     Name: .xdata
# CHECK:     Value: 0
# CHECK:     Section: .xdata (7)
# CHECK:     BaseType: Null (0x0)
# CHECK:     ComplexType: Null (0x0)
# CHECK:     StorageClass: Static (0x3)
# CHECK:     AuxSymbolCount: 1
# CHECK:     AuxSectionDef {
# CHECK:       Length: 8
# CHECK:       RelocationCount: 0
# CHECK:       LineNumberCount: 0
# CHECK:       Checksum: 0xCCAA009E
# CHECK:       Number: 4
# CHECK:       Selection: Associative (0x5)
# CHECK:       AssocSection: .text (4)
# CHECK:     }
# CHECK:   }
# CHECK:   Symbol {
# CHECK:     Name: .pdata
# CHECK:     Value: 0
# CHECK:     Section: .pdata (8)
# CHECK:     BaseType: Null (0x0)
# CHECK:     ComplexType: Null (0x0)
# CHECK:     StorageClass: Static (0x3)
# CHECK:     AuxSymbolCount: 1
# CHECK:     AuxSectionDef {
# CHECK:       Length: 12
# CHECK:       RelocationCount: 3
# CHECK:       LineNumberCount: 0
# CHECK:       Checksum: 0xD92012AC
# CHECK:       Number: 4
# CHECK:       Selection: Associative (0x5)
# CHECK:       AssocSection: .text (4)
# CHECK:     }
# CHECK:   }
# CHECK:   Symbol {
# CHECK:     Name: .pdata
# CHECK:     Value: 0
# CHECK:     Section: .pdata (9)
# CHECK:     BaseType: Null (0x0)
# CHECK:     ComplexType: Null (0x0)
# CHECK:     StorageClass: Static (0x3)
# CHECK:     AuxSymbolCount: 1
# CHECK:     AuxSectionDef {
# CHECK:       Length: 12
# CHECK:       RelocationCount: 3
# CHECK:       LineNumberCount: 0
# CHECK:       Checksum: 0xCCAA009E
# CHECK:       Number: 4
# CHECK:       Selection: Associative (0x5)
# CHECK:       AssocSection: .text (4)
# CHECK:     }
# CHECK:   }
# CHECK: ]
