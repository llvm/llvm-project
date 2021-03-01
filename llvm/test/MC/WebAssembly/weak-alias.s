# RUN: llvm-mc -triple=wasm32-unknown-unknown -filetype=obj < %s | obj2yaml | FileCheck --check-prefix=CHECK %s
# RUN: llvm-mc -triple=wasm32-unknown-unknown -mattr=+reference-types -filetype=obj < %s | obj2yaml | FileCheck --check-prefix=REF %s

# 'foo_alias()' is weak alias of function 'foo()'
# 'bar_alias' is weak alias of global variable 'bar'
# Generates two exports of the same function, one of them weak

foo:
  .hidden foo
  .globl  foo
  .functype foo () -> (i32)
  i32.const 0
  end_function

call_direct:
  .hidden call_direct
  .globl call_direct
  .functype call_direct () -> (i32)
  call foo
  end_function

call_alias:
  .hidden call_alias
  .globl call_alias
  .functype call_alias () -> (i32)
  call foo_alias
  end_function

call_direct_ptr:
  .hidden call_direct_ptr
  .globl call_direct_ptr
  .functype call_direct_ptr () -> (i32)
  i32.const 0
  i32.load direct_address
  call_indirect () -> (i32)
  end_function

call_alias_ptr:
  .hidden call_alias_ptr
  .globl call_alias_ptr
  .functype call_alias_ptr () -> (i32)
  i32.const 0
  i32.load alias_address
  call_indirect () -> (i32)
  end_function

.section .data.bar,"",@
bar:
  .int32   7
  .size    bar, 4
  .globl   bar
  .p2align 3

.section .data.direct_address,"",@
direct_address:
  .int32 foo
  .size  direct_address, 4
  .globl direct_address
  .p2align 3

.section .data.alias_address,"",@
alias_address:
  .int32 foo_alias
  .size  alias_address, 4
  .globl alias_address
  .p2align 3

# Define Aliases
.set foo_alias, foo
  .weak   foo_alias
  .type   foo_alias,@function
  .hidden foo_alias

.set bar_alias, bar
  .weak   bar_alias
  .hidden bar_alias

# CHECK:        - Type:            TYPE
# CHECK-NEXT:     Signatures:
# CHECK-NEXT:       - Index:           0
# CHECK-NEXT:         ParamTypes:      []
# CHECK-NEXT:         ReturnTypes:
# CHECK-NEXT:           - I32
# CHECK-NEXT:   - Type:            IMPORT
# CHECK-NEXT:     Imports:
# CHECK-NEXT:       - Module:          env
# CHECK-NEXT:         Field:           __linear_memory
# CHECK-NEXT:         Kind:            MEMORY
# CHECK-NEXT:         Memory:
# CHECK-NEXT:           Initial:         0x1
# CHECK-NEXT:       - Module:          env
# CHECK-NEXT:         Field:           __indirect_function_table
# CHECK-NEXT:         Kind:            TABLE
# CHECK-NEXT:         Table:
# CHECK-NEXT:           Index:           0
# CHECK-NEXT:           ElemType:        FUNCREF
# CHECK-NEXT:           Limits:
# CHECK-NEXT:             Initial:         0x1
# CHECK-NEXT:   - Type:            FUNCTION
# CHECK-NEXT:     FunctionTypes:   [ 0, 0, 0, 0, 0 ]
# CHECK-NEXT:   - Type:            ELEM
# CHECK-NEXT:     Segments:
# CHECK-NEXT:       - Offset:
# CHECK-NEXT:           Opcode:          I32_CONST
# CHECK-NEXT:           Value:           1
# CHECK-NEXT:         Functions:       [ 0 ]
# CHECK-NEXT:   - Type:            DATACOUNT
# CHECK-NEXT:     Count:           3
# CHECK-NEXT:   - Type:            CODE
# CHECK-NEXT:     Relocations:
# CHECK-NEXT:       - Type:            R_WASM_FUNCTION_INDEX_LEB
# CHECK-NEXT:         Index:           0
# CHECK-NEXT:         Offset:          0x9
# CHECK-NEXT:       - Type:            R_WASM_FUNCTION_INDEX_LEB
# CHECK-NEXT:         Index:           3
# CHECK-NEXT:         Offset:          0x12
# CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
# CHECK-NEXT:         Index:           5
# CHECK-NEXT:         Offset:          0x1E
# CHECK-NEXT:       - Type:            R_WASM_TYPE_INDEX_LEB
# CHECK-NEXT:         Index:           0
# CHECK-NEXT:         Offset:          0x24
# CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
# CHECK-NEXT:         Index:           7
# CHECK-NEXT:         Offset:          0x31
# CHECK-NEXT:       - Type:            R_WASM_TYPE_INDEX_LEB
# CHECK-NEXT:         Index:           0
# CHECK-NEXT:         Offset:          0x37
# CHECK-NEXT:     Functions:
# CHECK-NEXT:       - Index:           0
# CHECK-NEXT:         Locals:          []
# CHECK-NEXT:         Body:            41000B
# CHECK-NEXT:       - Index:           1
# CHECK-NEXT:         Locals:          []
# CHECK-NEXT:         Body:            1080808080000B
# CHECK-NEXT:       - Index:           2
# CHECK-NEXT:         Locals:          []
# CHECK-NEXT:         Body:            1080808080000B
# CHECK-NEXT:       - Index:           3
# CHECK-NEXT:         Locals:          []
# CHECK-NEXT:         Body:            410028028880808000118080808000000B
# CHECK-NEXT:       - Index:           4
# CHECK-NEXT:         Locals:          []
# CHECK-NEXT:         Body:            410028029080808000118080808000000B
# CHECK-NEXT:   - Type:            DATA
# CHECK-NEXT:     Relocations:
# CHECK-NEXT:       - Type:            R_WASM_TABLE_INDEX_I32
# CHECK-NEXT:         Index:           0
# CHECK-NEXT:         Offset:          0x13
# CHECK-NEXT:       - Type:            R_WASM_TABLE_INDEX_I32
# CHECK-NEXT:         Index:           3
# CHECK-NEXT:         Offset:          0x20
# CHECK-NEXT:     Segments:
# CHECK-NEXT:       - SectionOffset:   6
# CHECK-NEXT:         InitFlags:       0
# CHECK-NEXT:         Offset:
# CHECK-NEXT:           Opcode:          I32_CONST
# CHECK-NEXT:           Value:           0
# CHECK-NEXT:         Content:         '0700000000000000'
# CHECK-NEXT:       - SectionOffset:   19
# CHECK-NEXT:         InitFlags:       0
# CHECK-NEXT:         Offset:
# CHECK-NEXT:           Opcode:          I32_CONST
# CHECK-NEXT:           Value:           8
# CHECK-NEXT:         Content:         '0100000000000000'
# CHECK-NEXT:       - SectionOffset:   32
# CHECK-NEXT:         InitFlags:       0
# CHECK-NEXT:         Offset:
# CHECK-NEXT:           Opcode:          I32_CONST
# CHECK-NEXT:           Value:           16
# CHECK-NEXT:         Content:         '0100000000000000'
# CHECK-NEXT:   - Type:            CUSTOM
# CHECK-NEXT:     Name:            linking
# CHECK-NEXT:     Version:         2
# CHECK-NEXT:     SymbolTable:
# CHECK-NEXT:       - Index:           0
# CHECK-NEXT:         Kind:            FUNCTION
# CHECK-NEXT:         Name:            foo
# CHECK-NEXT:         Flags:           [ VISIBILITY_HIDDEN ]
# CHECK-NEXT:         Function:        0
# CHECK-NEXT:       - Index:           1
# CHECK-NEXT:         Kind:            FUNCTION
# CHECK-NEXT:         Name:            call_direct
# CHECK-NEXT:         Flags:           [ VISIBILITY_HIDDEN ]
# CHECK-NEXT:         Function:        1
# CHECK-NEXT:       - Index:           2
# CHECK-NEXT:         Kind:            FUNCTION
# CHECK-NEXT:         Name:            call_alias
# CHECK-NEXT:         Flags:           [ VISIBILITY_HIDDEN ]
# CHECK-NEXT:         Function:        2
# CHECK-NEXT:       - Index:           3
# CHECK-NEXT:         Kind:            FUNCTION
# CHECK-NEXT:         Name:            foo_alias
# CHECK-NEXT:         Flags:           [ BINDING_WEAK, VISIBILITY_HIDDEN, NO_STRIP ]
# CHECK-NEXT:         Function:        0
# CHECK-NEXT:       - Index:           4
# CHECK-NEXT:         Kind:            FUNCTION
# CHECK-NEXT:         Name:            call_direct_ptr
# CHECK-NEXT:         Flags:           [ VISIBILITY_HIDDEN ]
# CHECK-NEXT:         Function:        3
# CHECK-NEXT:       - Index:           5
# CHECK-NEXT:         Kind:            DATA
# CHECK-NEXT:         Name:            direct_address
# CHECK-NEXT:         Flags:           [  ]
# CHECK-NEXT:         Segment:         1
# CHECK-NEXT:         Size:            4
# CHECK-NEXT:       - Index:           6
# CHECK-NEXT:         Kind:            FUNCTION
# CHECK-NEXT:         Name:            call_alias_ptr
# CHECK-NEXT:         Flags:           [ VISIBILITY_HIDDEN ]
# CHECK-NEXT:         Function:        4
# CHECK-NEXT:       - Index:           7
# CHECK-NEXT:         Kind:            DATA
# CHECK-NEXT:         Name:            alias_address
# CHECK-NEXT:         Flags:           [  ]
# CHECK-NEXT:         Segment:         2
# CHECK-NEXT:         Size:            4
# CHECK-NEXT:       - Index:           8
# CHECK-NEXT:         Kind:            DATA
# CHECK-NEXT:         Name:            bar
# CHECK-NEXT:         Flags:           [  ]
# CHECK-NEXT:         Segment:         0
# CHECK-NEXT:         Size:            4
# CHECK-NEXT:       - Index:           9
# CHECK-NEXT:         Kind:            DATA
# CHECK-NEXT:         Name:            bar_alias
# CHECK-NEXT:         Flags:           [ BINDING_WEAK, VISIBILITY_HIDDEN, NO_STRIP ]
# CHECK-NEXT:         Segment:         0
# CHECK-NEXT:         Size:            4
# CHECK-NEXT:     SegmentInfo:
# CHECK-NEXT:       - Index:           0
# CHECK-NEXT:         Name:            .data.bar
# CHECK-NEXT:         Alignment:       3
# CHECK-NEXT:         Flags:           [  ]
# CHECK-NEXT:       - Index:           1
# CHECK-NEXT:         Name:            .data.direct_address
# CHECK-NEXT:         Alignment:       3
# CHECK-NEXT:         Flags:           [  ]
# CHECK-NEXT:       - Index:           2
# CHECK-NEXT:         Name:            .data.alias_address
# CHECK-NEXT:         Alignment:       3
# CHECK-NEXT:         Flags:           [  ]
# CHECK-NEXT: ...

# REF:        - Type:            TYPE
# REF-NEXT:     Signatures:
# REF-NEXT:       - Index:           0
# REF-NEXT:         ParamTypes:      []
# REF-NEXT:         ReturnTypes:
# REF-NEXT:           - I32
# REF-NEXT:   - Type:            IMPORT
# REF-NEXT:     Imports:
# REF-NEXT:       - Module:          env
# REF-NEXT:         Field:           __linear_memory
# REF-NEXT:         Kind:            MEMORY
# REF-NEXT:         Memory:
# REF-NEXT:           Initial:         0x1
# REF-NEXT:       - Module:          env
# REF-NEXT:         Field:           __indirect_function_table
# REF-NEXT:         Kind:            TABLE
# REF-NEXT:         Table:
# REF-NEXT:           Index:           0
# REF-NEXT:           ElemType:        FUNCREF
# REF-NEXT:           Limits:
# REF-NEXT:             Initial:         0x1
# REF-NEXT:   - Type:            FUNCTION
# REF-NEXT:     FunctionTypes:   [ 0, 0, 0, 0, 0 ]
# REF-NEXT:   - Type:            ELEM
# REF-NEXT:     Segments:
# REF-NEXT:       - Offset:
# REF-NEXT:           Opcode:          I32_CONST
# REF-NEXT:           Value:           1
# REF-NEXT:         Functions:       [ 0 ]
# REF-NEXT:   - Type:            DATACOUNT
# REF-NEXT:     Count:           3
# REF-NEXT:   - Type:            CODE
# REF-NEXT:     Relocations:
# REF-NEXT:       - Type:            R_WASM_FUNCTION_INDEX_LEB
# REF-NEXT:         Index:           0
# REF-NEXT:         Offset:          0x9
# REF-NEXT:       - Type:            R_WASM_FUNCTION_INDEX_LEB
# REF-NEXT:         Index:           3
# REF-NEXT:         Offset:          0x12
# REF-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
# REF-NEXT:         Index:           5
# REF-NEXT:         Offset:          0x1E
# REF-NEXT:       - Type:            R_WASM_TYPE_INDEX_LEB
# REF-NEXT:         Index:           0
# REF-NEXT:         Offset:          0x24
# REF-NEXT:       - Type:            R_WASM_TABLE_NUMBER_LEB
# REF-NEXT:         Index:           6
# REF-NEXT:         Offset:          0x29
# REF-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
# REF-NEXT:         Index:           8
# REF-NEXT:         Offset:          0x35
# REF-NEXT:       - Type:            R_WASM_TYPE_INDEX_LEB
# REF-NEXT:         Index:           0
# REF-NEXT:         Offset:          0x3B
# REF-NEXT:       - Type:            R_WASM_TABLE_NUMBER_LEB
# REF-NEXT:         Index:           6
# REF-NEXT:         Offset:          0x40
# REF-NEXT:     Functions:
# REF-NEXT:       - Index:           0
# REF-NEXT:         Locals:          []
# REF-NEXT:         Body:            41000B
# REF-NEXT:       - Index:           1
# REF-NEXT:         Locals:          []
# REF-NEXT:         Body:            1080808080000B
# REF-NEXT:       - Index:           2
# REF-NEXT:         Locals:          []
# REF-NEXT:         Body:            1080808080000B
# REF-NEXT:       - Index:           3
# REF-NEXT:         Locals:          []
# REF-NEXT:         Body:            41002802888080800011808080800080808080000B
# REF-NEXT:       - Index:           4
# REF-NEXT:         Locals:          []
# REF-NEXT:         Body:            41002802908080800011808080800080808080000B
# REF-NEXT:   - Type:            DATA
# REF-NEXT:     Relocations:
# REF-NEXT:       - Type:            R_WASM_TABLE_INDEX_I32
# REF-NEXT:         Index:           0
# REF-NEXT:         Offset:          0x13
# REF-NEXT:       - Type:            R_WASM_TABLE_INDEX_I32
# REF-NEXT:         Index:           3
# REF-NEXT:         Offset:          0x20
# REF-NEXT:     Segments:
# REF-NEXT:       - SectionOffset:   6
# REF-NEXT:         InitFlags:       0
# REF-NEXT:         Offset:
# REF-NEXT:           Opcode:          I32_CONST
# REF-NEXT:           Value:           0
# REF-NEXT:         Content:         '0700000000000000'
# REF-NEXT:       - SectionOffset:   19
# REF-NEXT:         InitFlags:       0
# REF-NEXT:         Offset:
# REF-NEXT:           Opcode:          I32_CONST
# REF-NEXT:           Value:           8
# REF-NEXT:         Content:         '0100000000000000'
# REF-NEXT:       - SectionOffset:   32
# REF-NEXT:         InitFlags:       0
# REF-NEXT:         Offset:
# REF-NEXT:           Opcode:          I32_CONST
# REF-NEXT:           Value:           16
# REF-NEXT:         Content:         '0100000000000000'
# REF-NEXT:   - Type:            CUSTOM
# REF-NEXT:     Name:            linking
# REF-NEXT:     Version:         2
# REF-NEXT:     SymbolTable:
# REF-NEXT:       - Index:           0
# REF-NEXT:         Kind:            FUNCTION
# REF-NEXT:         Name:            foo
# REF-NEXT:         Flags:           [ VISIBILITY_HIDDEN ]
# REF-NEXT:         Function:        0
# REF-NEXT:       - Index:           1
# REF-NEXT:         Kind:            FUNCTION
# REF-NEXT:         Name:            call_direct
# REF-NEXT:         Flags:           [ VISIBILITY_HIDDEN ]
# REF-NEXT:         Function:        1
# REF-NEXT:       - Index:           2
# REF-NEXT:         Kind:            FUNCTION
# REF-NEXT:         Name:            call_alias
# REF-NEXT:         Flags:           [ VISIBILITY_HIDDEN ]
# REF-NEXT:         Function:        2
# REF-NEXT:       - Index:           3
# REF-NEXT:         Kind:            FUNCTION
# REF-NEXT:         Name:            foo_alias
# REF-NEXT:         Flags:           [ BINDING_WEAK, VISIBILITY_HIDDEN, NO_STRIP ]
# REF-NEXT:         Function:        0
# REF-NEXT:       - Index:           4
# REF-NEXT:         Kind:            FUNCTION
# REF-NEXT:         Name:            call_direct_ptr
# REF-NEXT:         Flags:           [ VISIBILITY_HIDDEN ]
# REF-NEXT:         Function:        3
# REF-NEXT:       - Index:           5
# REF-NEXT:         Kind:            DATA
# REF-NEXT:         Name:            direct_address
# REF-NEXT:         Flags:           [  ]
# REF-NEXT:         Segment:         1
# REF-NEXT:         Size:            4
# REF-NEXT:       - Index:           6
# REF-NEXT:         Kind:            TABLE
# REF-NEXT:         Name:            __indirect_function_table
# REF-NEXT:         Flags:           [ UNDEFINED, NO_STRIP ]
# REF-NEXT:         Table:           0
# REF-NEXT:       - Index:           7
# REF-NEXT:         Kind:            FUNCTION
# REF-NEXT:         Name:            call_alias_ptr
# REF-NEXT:         Flags:           [ VISIBILITY_HIDDEN ]
# REF-NEXT:         Function:        4
# REF-NEXT:       - Index:           8
# REF-NEXT:         Kind:            DATA
# REF-NEXT:         Name:            alias_address
# REF-NEXT:         Flags:           [  ]
# REF-NEXT:         Segment:         2
# REF-NEXT:         Size:            4
# REF-NEXT:       - Index:           9
# REF-NEXT:         Kind:            DATA
# REF-NEXT:         Name:            bar
# REF-NEXT:         Flags:           [  ]
# REF-NEXT:         Segment:         0
# REF-NEXT:         Size:            4
# REF-NEXT:       - Index:           10
# REF-NEXT:         Kind:            DATA
# REF-NEXT:         Name:            bar_alias
# REF-NEXT:         Flags:           [ BINDING_WEAK, VISIBILITY_HIDDEN, NO_STRIP ]
# REF-NEXT:         Segment:         0
# REF-NEXT:         Size:            4
# REF-NEXT:     SegmentInfo:
# REF-NEXT:       - Index:           0
# REF-NEXT:         Name:            .data.bar
# REF-NEXT:         Alignment:       3
# REF-NEXT:         Flags:           [  ]
# REF-NEXT:       - Index:           1
# REF-NEXT:         Name:            .data.direct_address
# REF-NEXT:         Alignment:       3
# REF-NEXT:         Flags:           [  ]
# REF-NEXT:       - Index:           2
# REF-NEXT:         Name:            .data.alias_address
# REF-NEXT:         Alignment:       3
# REF-NEXT:         Flags:           [  ]
# REF-NEXT: ...

# CHECK-SYMS: SYMBOL TABLE:
# CHECK-SYMS-NEXT: 00000001 g     F CODE	.hidden foo
# CHECK-SYMS-NEXT: 00000006 g     F CODE	.hidden call_direct
# CHECK-SYMS-NEXT: 0000000f g     F CODE	.hidden call_alias
# CHECK-SYMS-NEXT: 00000001 gw    F CODE	.hidden foo_alias
# CHECK-SYMS-NEXT: 00000018 g     F CODE	.hidden call_direct_ptr
# CHECK-SYMS-NEXT: 00000008 g     O DATA	direct_address
# CHECK-SYMS-NEXT: 0000002b g     F CODE	.hidden call_alias_ptr
# CHECK-SYMS-NEXT: 00000010 g     O DATA	alias_address
# CHECK-SYMS-NEXT: 00000000 g     O DATA	bar
# CHECK-SYMS-NEXT: 00000000 gw    O DATA	.hidden bar_alias
