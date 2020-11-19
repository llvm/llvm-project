# RUN: llvm-mc -triple=wasm32-unknown-unknown -mattr=+reference-types < %s | FileCheck %s
# RUN: llvm-mc -show-encoding -triple=wasm32-unknown-unknown -mattr=+reference-types < %s | FileCheck -check-prefix=ENC %s
# RUN: llvm-mc -triple=wasm32-unknown-unknown -mattr=+reference-types -filetype=obj < %s | obj2yaml | FileCheck -check-prefix=BIN %s

# Creating several empty tables

# CHECK:         .tabletype foo, externref
# CHECK: foo:
    .tabletype foo, externref
foo:


#      CHECK: bar:
# CHECK-NEXT:         .tabletype bar, funcref
bar:
    .tabletype bar, funcref

table1:
    .tabletype table1, funcref
table2:
    .tabletype table2, funcref

# Table instructions

#      CHECK: copy_tables:
# CHECK-NEXT:         .functype	copy_tables (i32, i32) -> ()
# CHECK-NEXT:	local.get	0
# CHECK-NEXT:	local.get	1
#      CHECK:	table.size	table1
#      CHECK:	table.copy	table1, table2
# CHECK-NEXT:	end_function
# CHECK-NEXT:.Ltmp0:
# CHECK-NEXT:	.size	copy_tables, .Ltmp0-copy_tables
copy_tables:
    .functype copy_tables (i32, i32) -> ()
    local.get 0
    local.get 1

    # ENC: table.size	table1                  # encoding: [0xfc,0x10,0x80'A',0x80'A',0x80'A',0x80'A',A]
    table.size table1

    # ENC: table.copy	table1, table2          # encoding: [0xfc,0x0e,0x80'A',0x80'A',0x80'A',0x80'A',A,0x80'B',0x80'B',0x80'B',0x80'B',B]
    table.copy table1, table2
    end_function

#      CHECK: table_get:
# CHECK-NEXT:	.functype	table_get (i32) -> (externref)
# CHECK-NEXT:	local.get	0
#      CHECK:	table.get	foo
# CHECK-NEXT:	end_function
# CHECK-NEXT: .Ltmp1:
# CHECK-NEXT:	.size	table_get, .Ltmp1-table_get
table_get:
    .functype table_get (i32) -> (externref)
    local.get 0

    # ENC: table.get	foo                     # encoding: [0x25,0x80'A',0x80'A',0x80'A',0x80'A',A]
    table.get foo
    end_function

#      CHECK: table_set:
# CHECK-NEXT:	.functype	table_set (i32, externref) -> ()
# CHECK-NEXT:	local.get	0
# CHECK-NEXT:	local.get	1
#      CHECK:	table.set	foo
# CHECK-NEXT:	end_function
# CHECK-NEXT: .Ltmp2:
# CHECK-NEXT:	.size	table_set, .Ltmp2-table_set
table_set:
    .functype table_set (i32, externref) -> ()
    local.get 0
    local.get 1

    # ENC: table.set	foo                     # encoding: [0x26,0x80'A',0x80'A',0x80'A',0x80'A',A]
    table.set foo
    end_function

#      CHECK: table_grow:
# CHECK-NEXT:	.functype	table_grow (i32) -> (i32)
# CHECK-NEXT:	i32.const	0
# CHECK-NEXT:	table.get	foo
# CHECK-NEXT:	local.get	0
#      CHECK:	table.grow	foo
# CHECK-NEXT:	local.get	0
# CHECK-NEXT:	i32.add
# CHECK-NEXT:	end_function
# CHECK-NEXT: .Ltmp3:
# CHECK-NEXT:	.size	table_grow, .Ltmp3-table_grow
table_grow:
    .functype table_grow (i32) -> (i32)
    i32.const 0
    table.get foo
    local.get 0

    # ENC: table.grow	foo                     # encoding: [0xfc,0x0f,0x80'A',0x80'A',0x80'A',0x80'A',A]
    table.grow foo
    local.get 0
    i32.add
    end_function

#      CHECK: table_fill:
# CHECK-NEXT:	.functype	table_fill (i32, i32) -> ()
# CHECK-NEXT:	local.get	0
# CHECK-NEXT:	i32.const	0
# CHECK-NEXT:	table.get	table1
# CHECK-NEXT:	local.get	1
#      CHECK:	table.fill	table1
# CHECK-NEXT:	end_function
# CHECK-NEXT: .Ltmp4:
# CHECK-NEXT:	.size	table_fill, .Ltmp4-table_fill
table_fill:
    .functype table_fill (i32, i32) -> ()
    local.get 0
    i32.const 0
    table.get table1
    local.get 1

    # ENC: table.fill	table1                  # encoding: [0xfc,0x11,0x80'A',0x80'A',0x80'A',0x80'A',A]
    table.fill table1
    end_function

#      BIN:  - Type:            TABLE
# BIN-NEXT:    Tables:
# BIN-NEXT:      - Index:           1
# BIN-NEXT:        ElemType:        EXTERNREF
# BIN-NEXT:        Limits:
# BIN-NEXT:          Initial:         0x0
# BIN-NEXT:      - Index:           2
# BIN-NEXT:        ElemType:        FUNCREF
# BIN-NEXT:        Limits:
# BIN-NEXT:          Initial:         0x0

#      BIN:  - Type:            CODE
# BIN-NEXT:    Relocations:
# BIN-NEXT:      - Type:            R_WASM_TABLE_NUMBER_LEB
# BIN-NEXT:        Index:           2
# BIN-NEXT:        Offset:          0x9
# BIN-NEXT:      - Type:            R_WASM_TABLE_NUMBER_LEB
# BIN-NEXT:        Index:           2
# BIN-NEXT:        Offset:          0x10
# BIN-NEXT:      - Type:            R_WASM_TABLE_NUMBER_LEB
# BIN-NEXT:        Index:           3
# BIN-NEXT:        Offset:          0x15
# BIN-NEXT:      - Type:            R_WASM_TABLE_NUMBER_LEB
# BIN-NEXT:        Index:           0
# BIN-NEXT:        Offset:          0x20
# BIN-NEXT:      - Type:            R_WASM_TABLE_NUMBER_LEB
# BIN-NEXT:        Index:           0
# BIN-NEXT:        Offset:          0x2D
# BIN-NEXT:      - Type:            R_WASM_TABLE_NUMBER_LEB
# BIN-NEXT:        Index:           0
# BIN-NEXT:        Offset:          0x38
# BIN-NEXT:      - Type:            R_WASM_TABLE_NUMBER_LEB
# BIN-NEXT:        Index:           0
# BIN-NEXT:        Offset:          0x41
# BIN-NEXT:      - Type:            R_WASM_TABLE_NUMBER_LEB
# BIN-NEXT:        Index:           2
# BIN-NEXT:        Offset:          0x51
# BIN-NEXT:      - Type:            R_WASM_TABLE_NUMBER_LEB
# BIN-NEXT:        Index:           2
# BIN-NEXT:        Offset:          0x5A
# BIN-NEXT:    Functions:
# BIN-NEXT:      - Index:           0
# BIN-NEXT:        Locals:          []
# BIN-NEXT:        Body:            20002001FC108380808000FC0E838080800084808080000B
# BIN-NEXT:      - Index:           1
# BIN-NEXT:        Locals:          []
# BIN-NEXT:        Body:            20002581808080000B
# BIN-NEXT:      - Index:           2
# BIN-NEXT:        Locals:          []
# BIN-NEXT:        Body:            200020012681808080000B
# BIN-NEXT:      - Index:           3
# BIN-NEXT:        Locals:          []
# BIN-NEXT:        Body:            41002581808080002000FC0F818080800020006A0B
# BIN-NEXT:      - Index:           4
# BIN-NEXT:        Locals:          []
# BIN-NEXT:        Body:            200041002583808080002001FC1183808080000B

#      BIN:  - Type:            CUSTOM
# BIN-NEXT:    Name:            linking
# BIN-NEXT:    Version:         2
# BIN-NEXT:    SymbolTable:
# BIN-NEXT:      - Index:           0
# BIN-NEXT:        Kind:            TABLE
# BIN-NEXT:        Name:            foo
# BIN-NEXT:        Flags:           [ BINDING_LOCAL ]
# BIN-NEXT:        Table:           1
# BIN-NEXT:      - Index:           1
# BIN-NEXT:        Kind:            TABLE
# BIN-NEXT:        Name:            bar
# BIN-NEXT:        Flags:           [ BINDING_LOCAL ]
# BIN-NEXT:        Table:           2
