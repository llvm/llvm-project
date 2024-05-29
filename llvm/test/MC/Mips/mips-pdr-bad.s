# RUN: not llvm-mc %s -triple=mips -mcpu=mips32r2 2>%t1
# RUN: FileCheck %s < %t1 -check-prefix=ASM

        .text
        
        .ent # ASM: :[[@LINE]]:14: error: expected identifier after .ent
        .ent bar, # ASM: :[[@LINE]]:19: error: expected number after comma
        .ent foo, bar # ASM: :[[@LINE]]:23: error: expected an absolute expression after comma
        .ent foo, 5, bar # ASM: :[[@LINE]]:20: error: unexpected token, expected end of statement
 
        .frame # ASM: :[[@LINE]]:16: error: expected stack register
        .frame bar # ASM: :[[@LINE]]:16: error: expected stack register
        .frame $f1, 8, # ASM: :[[@LINE]]:16: error: expected general purpose register
        .frame $sp # ASM: :[[@LINE]]:20: error: unexpected token, expected comma
        .frame $sp, # ASM: :[[@LINE]]:21: error: expected frame size value
        .frame $sp, bar # ASM: :[[@LINE]]:25: error: frame size not an absolute expression
        .frame $sp, 8 # ASM: :[[@LINE]]:23: error: unexpected token, expected comma
        .frame $sp, 8, # ASM: :[[@LINE]]:24: error: expected return register
        .frame $sp, 8, $f1 # ASM: :[[@LINE]]:24: error: expected general purpose register
        .frame $sp, 8, $ra, foo # ASM: :[[@LINE]]:27: error: unexpected token, expected end of statement

        .mask  # ASM: :[[@LINE]]:16: error: expected bitmask value
        .mask foo # ASM: :[[@LINE]]:19: error: bitmask not an absolute expression
        .mask 0x80000000 # ASM: :[[@LINE]]:26: error: unexpected token, expected comma
        .mask 0x80000000, # ASM: :[[@LINE]]:27: error: expected frame offset value
        .mask 0x80000000, foo # ASM: :[[@LINE]]:31: error: frame offset not an absolute expression
        .mask 0x80000000, -4, bar # ASM: :[[@LINE]]:29: error: unexpected token, expected end of statement
        
        .fmask  # ASM: :[[@LINE]]:17: error: expected bitmask value
        .fmask foo # ASM: :[[@LINE]]:20: error: bitmask not an absolute expression
        .fmask 0x80000000 # ASM: :[[@LINE]]:27: error: unexpected token, expected comma
        .fmask 0x80000000, # ASM: :[[@LINE]]:28: error: expected frame offset value
        .fmask 0x80000000, foo # ASM: :[[@LINE]]:32: error: frame offset not an absolute expression
        .fmask 0x80000000, -4, bar # ASM: :[[@LINE]]:30: error: unexpected token, expected end of statement

        .end # ASM: :[[@LINE]]:14: error: expected identifier after .end
        .ent _local_foo_bar
        .end _local_foo_bar, foo # ASM: :[[@LINE]]:28: error: unexpected token, expected end of statement
        .end _local_foo_bar
        .end _local_foo # ASM: :[[@LINE]]:25: error: .end used without .ent
        .ent _local_foo, 2
        .end _local_foo_bar # ASM: :[[@LINE]]:29: error: .end symbol does not match .ent symbol
