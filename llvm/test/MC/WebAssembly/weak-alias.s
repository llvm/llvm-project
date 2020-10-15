# RUN: llvm-mc -triple=wasm32-unknown-unknown -filetype=obj -o %t.o < %s
# RUN: obj2yaml %t.o | FileCheck %s

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
# CHECK-NEXT:         ParamTypes:
# CHECK-NEXT:         ReturnTypes:
# CHECK-NEXT:           - I32
# CHECK-NEXT:   - Type:            IMPORT
# CHECK-NEXT:     Imports:
# CHECK-NEXT:       - Module:          env
# CHECK-NEXT:         Field:           __linear_memory
# CHECK-NEXT:         Kind:            MEMORY
# CHECK-NEXT:         Memory:
# CHECK-NEXT:           Initial:         0x00000001
# CHECK-NEXT:       - Module:          env
# CHECK-NEXT:         Field:           __indirect_function_table
# CHECK-NEXT:         Kind:            TABLE
# CHECK-NEXT:         Table:
# CHECK-NEXT:           Index:           0
# CHECK-NEXT:           ElemType:        FUNCREF
# CHECK-NEXT:           Limits:
# CHECK-NEXT:             Initial:         0x00000001
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
# CHECK-NEXT:         Offset:          0x00000009
# CHECK-NEXT:       - Type:            R_WASM_FUNCTION_INDEX_LEB
# CHECK-NEXT:         Index:           3
# CHECK-NEXT:         Offset:          0x00000012
# CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
# CHECK-NEXT:         Index:           5
# CHECK-NEXT:         Offset:          0x0000001E
# CHECK-NEXT:       - Type:            R_WASM_TYPE_INDEX_LEB
# CHECK-NEXT:         Index:           0
# CHECK-NEXT:         Offset:          0x00000024
# CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
# CHECK-NEXT:         Index:           7
# CHECK-NEXT:         Offset:          0x00000031
# CHECK-NEXT:       - Type:            R_WASM_TYPE_INDEX_LEB
# CHECK-NEXT:         Index:           0
# CHECK-NEXT:         Offset:          0x00000037
# CHECK-NEXT:     Functions:
# CHECK-NEXT:       - Index:           0
# CHECK-NEXT:         Locals:
# CHECK-NEXT:         Body:            41000B
# CHECK-NEXT:       - Index:           1
# CHECK-NEXT:         Locals:
# CHECK-NEXT:         Body:            1080808080000B
# CHECK-NEXT:       - Index:           2
# CHECK-NEXT:         Locals:
# CHECK-NEXT:         Body:            1080808080000B
# CHECK-NEXT:       - Index:           3
# CHECK-NEXT:         Locals:
# CHECK-NEXT:         Body:            410028028880808000118080808000000B
# CHECK-NEXT:       - Index:           4
# CHECK-NEXT:         Locals:
# CHECK-NEXT:         Body:            410028029080808000118080808000000B
# CHECK-NEXT:   - Type:            DATA
# CHECK-NEXT:     Relocations:
# CHECK-NEXT:       - Type:            R_WASM_TABLE_INDEX_I32
# CHECK-NEXT:         Index:           0
# CHECK-NEXT:         Offset:          0x00000013
# CHECK-NEXT:       - Type:            R_WASM_TABLE_INDEX_I32
# CHECK-NEXT:         Index:           3
# CHECK-NEXT:         Offset:          0x00000020
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
# CHECK-NEXT:         Flags:           [ ]
# CHECK-NEXT:       - Index:           1
# CHECK-NEXT:         Name:            .data.direct_address
# CHECK-NEXT:         Alignment:       3
# CHECK-NEXT:         Flags:           [ ]
# CHECK-NEXT:       - Index:           2
# CHECK-NEXT:         Name:            .data.alias_address
# CHECK-NEXT:         Alignment:       3
# CHECK-NEXT:         Flags:           [ ]
# CHECK-NEXT: ...

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
