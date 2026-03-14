# Supplies a global definition of an absolute symbol, plus a global symbol
# containing the absolute symbol's value.

        .section        __TEXT,__text,regular,pure_instructions
        .build_version macos, 12, 0
        .section        __DATA,__data

        .globl _AbsoluteSymDef
_AbsoluteSymDef = 0x01234567

        .globl  _GlobalAbsoluteSymDefValue
        .p2align        2
_GlobalAbsoluteSymDefValue:
        .long _AbsoluteSymDef

.subsections_via_symbols
