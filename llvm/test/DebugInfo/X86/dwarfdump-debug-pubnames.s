# RUN: llvm-mc -triple x86_64-unknown-linux %s -filetype=obj -o - | \
# RUN:   llvm-dwarfdump -debug-pubnames - | \
# RUN:   FileCheck %s

# CHECK: .debug_pubnames contents:
# CHECK-NEXT: length = 0x00000032
# CHECK-SAME: version = 0x0002
# CHECK-SAME: unit_offset = 0x1122334455667788
# CHECK-SAME: unit_size = 0x1100220033004400
# CHECK-NEXT: Offset     Name
# CHECK-NEXT: 0xaa01aaaabbbbbbbb "foo"
# CHECK-NEXT: 0xaa02aaaabbbbbbbb "bar"

    .section .debug_pubnames,"",@progbits
    .long 0xffffffff            # DWARF64 mark
    .quad .Lend - .Lversion     # Unit Length
.Lversion:
    .short 2                    # Version
    .quad 0x1122334455667788    # Debug Info Offset
    .quad 0x1100220033004400    # Debug Info Length
    .quad 0xaa01aaaabbbbbbbb    # Tuple0: Offset
    .asciz "foo"                #         Name
    .quad 0xaa02aaaabbbbbbbb    # Tuple1: Offset
    .asciz "bar"                #         Name
    .quad 0                     # Terminator
.Lend:
