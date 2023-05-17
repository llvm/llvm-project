# Supplies an internal definition of an absolute symbol, plus a global symbol
# containing the absolute symbol's value.

        .section        __TEXT,__text,regular,pure_instructions
        .build_version macos, 12, 0
        .section        __DATA,__data

_AbsoluteSymDef = 0x89ABCDEF

        .globl  _LocalAbsoluteSymDefValue
        .p2align        2
_LocalAbsoluteSymDefValue:
        .long _AbsoluteSymDef

.subsections_via_symbols
