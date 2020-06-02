# RUN: llvm-mc -triple x86_64-unknown-linux %s -filetype=obj -o - | \
# RUN:   llvm-dwarfdump -debug-pubnames - | \
# RUN:   FileCheck %s

# CHECK: .debug_pubnames contents:
# CHECK-NEXT: length = 0x0000000000000032
# CHECK-SAME: format = DWARF64
# CHECK-SAME: version = 0x0002
# CHECK-SAME: unit_offset = 0x0000112233445566
# CHECK-SAME: unit_size = 0x0000110022003300
# CHECK-NEXT: Offset     Name
# CHECK-NEXT: 0x0000aa01aaaabbbb "foo"
# CHECK-NEXT: 0x0000aa02aaaabbbb "bar"

    .section .debug_pubnames,"",@progbits
    .long 0xffffffff            # DWARF64 mark
    .quad .Lend - .Lversion     # Unit Length
.Lversion:
    .short 2                    # Version
    .quad 0x112233445566        # Debug Info Offset
    .quad 0x110022003300        # Debug Info Length
    .quad 0xaa01aaaabbbb        # Tuple0: Offset
    .asciz "foo"                #         Name
    .quad 0xaa02aaaabbbb        # Tuple1: Offset
    .asciz "bar"                #         Name
    .quad 0                     # Terminator
.Lend:
