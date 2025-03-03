# RUN: llvm-mc -triple=wasm32-unknown-unknown < %s | FileCheck %s
# RUN: llvm-mc -triple=wasm32-unknown-unknown -filetype=obj < %s | obj2yaml | FileCheck -check-prefix=BIN %s

# Tests creating an accessing actual wasm globals

.globl read_global
.globl write_global
.globaltype foo_global, i32
.globaltype global2, i64, immutable
.globaltype global3, f32
.globaltype global4, f64

read_global:
  .functype read_global () -> (i32)
  global.get foo_global
  end_function

write_global:
  .functype write_global (i32, i64, f32, f64) -> ()
  local.get 0
  global.set foo_global
  local.get 1
  global.set global2
  local.get 2
  global.set global3
  local.get 3
  global.set global4
  end_function

foo_global:
global2:
global3:
global4:

# CHECK: .globl  read_global
# CHECK: .globl  write_global
# CHECK: .globaltype foo_global, i32
# CHECK: foo_global:

#      BIN: - Type:            GLOBAL
# BIN-NEXT:   Globals:
# BIN-NEXT:     - Index:           0
# BIN-NEXT:       Type:            I32
# BIN-NEXT:       Mutable:         true
# BIN-NEXT:       InitExpr:
# BIN-NEXT:         Opcode:          I32_CONST
# BIN-NEXT:         Value:           0
# BIN-NEXT:     - Index:           1
# BIN-NEXT:       Type:            I64
# BIN-NEXT:       Mutable:         false
# BIN-NEXT:       InitExpr:
# BIN-NEXT:         Opcode:          I64_CONST
# BIN-NEXT:         Value:           0

#      BIN:  - Type:            CUSTOM
# BIN-NEXT:    Name:            linking
# BIN-NEXT:    Version:         2
# BIN-NEXT:    SymbolTable:
# BIN-NEXT:      - Index:           0
# BIN-NEXT:        Kind:            FUNCTION
# BIN-NEXT:        Name:            read_global
# BIN-NEXT:        Flags:           [  ]
# BIN-NEXT:        Function:        0
# BIN-NEXT:      - Index:           1
# BIN-NEXT:        Kind:            FUNCTION
# BIN-NEXT:        Name:            write_global
# BIN-NEXT:        Flags:           [  ]
# BIN-NEXT:        Function:        1
# BIN-NEXT:      - Index:           2
# BIN-NEXT:        Kind:            GLOBAL
# BIN-NEXT:        Name:            foo_global
# BIN-NEXT:        Flags:           [ BINDING_LOCAL ]
# BIN-NEXT:        Global:          0
# BIN-NEXT:      - Index:           3
# BIN-NEXT:        Kind:            GLOBAL
# BIN-NEXT:        Name:            global2
# BIN-NEXT:        Flags:           [ BINDING_LOCAL ]
# BIN-NEXT:        Global:          1
# BIN-NEXT:      - Index:           4
# BIN-NEXT:        Kind:            GLOBAL
# BIN-NEXT:        Name:            global3
# BIN-NEXT:        Flags:           [ BINDING_LOCAL ]
# BIN-NEXT:        Global:          2
# BIN-NEXT:      - Index:           5
# BIN-NEXT:        Kind:            GLOBAL
# BIN-NEXT:        Name:            global4
# BIN-NEXT:        Flags:           [ BINDING_LOCAL ]
# BIN-NEXT:        Global:          3
