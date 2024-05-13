# RUN: llvm-mc -triple=wasm32-unknown-unknown -filetype=obj %s -o - | llvm-readobj -r --expand-relocs - | FileCheck %s
# RUN: llvm-mc -triple=wasm32-unknown-unknown -mattr=+reference-types -filetype=obj %s -o - | llvm-readobj -r --expand-relocs - | FileCheck --check-prefix=REF %s

# External functions
.functype c () -> (i32)
.functype d () -> (i32)

.globl  f1
.globl  a
.globl  b

f1:
  .functype f1 () -> (i32)

  # Call functions at `a` and `b` indirectly.
  i32.const 0
  i32.load  a - 10
  call_indirect  () -> (i64)
  drop

  i32.const 0
  i32.load  b + 20
  call_indirect  () -> (i32)
  drop

  # Call functions `c` and `d` directly
  call  c
  drop
  call  d
  end_function

# Pointers to functions of two different types
.section  .data.a,"",@
.p2align  3
a:
  .int32  5
  .size a, 4

.section  .data.b,"",@
.p2align  3
b:
  .int32  7
  .size b, 4

# CHECK: Format: WASM
# CHECK: Relocations [
# CHECK-NEXT:   Section (5) CODE {
# CHECK-NEXT:     Relocation {
# CHECK-NEXT:       Type: R_WASM_MEMORY_ADDR_LEB (3)
# CHECK-NEXT:       Offset: 0x7
# CHECK-NEXT:       Symbol: a
# CHECK-NEXT:       Addend: -10
# CHECK-NEXT:     }
# CHECK-NEXT:     Relocation {
# CHECK-NEXT:       Type: R_WASM_TYPE_INDEX_LEB (6)
# CHECK-NEXT:       Offset: 0xD
# CHECK-NEXT:       Index: 0x1
# CHECK-NEXT:     }
# CHECK-NEXT:     Relocation {
# CHECK-NEXT:       Type: R_WASM_MEMORY_ADDR_LEB (3)
# CHECK-NEXT:       Offset: 0x18
# CHECK-NEXT:       Symbol: b
# CHECK-NEXT:       Addend: 20
# CHECK-NEXT:     }
# CHECK-NEXT:     Relocation {
# CHECK-NEXT:       Type: R_WASM_TYPE_INDEX_LEB (6)
# CHECK-NEXT:       Offset: 0x1E
# CHECK-NEXT:       Index: 0x0
# CHECK-NEXT:     }
# CHECK-NEXT:     Relocation {
# CHECK-NEXT:       Type: R_WASM_FUNCTION_INDEX_LEB (0)
# CHECK-NEXT:       Offset: 0x26
# CHECK-NEXT:       Symbol: c
# CHECK-NEXT:     }
# CHECK-NEXT:     Relocation {
# CHECK-NEXT:       Type: R_WASM_FUNCTION_INDEX_LEB (0)
# CHECK-NEXT:       Offset: 0x2D
# CHECK-NEXT:       Symbol: d
# CHECK-NEXT:     }
# CHECK-NEXT:   }
# CHECK-NEXT: ]

# REF: Format: WASM
# REF: Relocations [
# REF-NEXT:   Section (5) CODE {
# REF-NEXT:     Relocation {
# REF-NEXT:       Type: R_WASM_MEMORY_ADDR_LEB (3)
# REF-NEXT:       Offset: 0x7
# REF-NEXT:       Symbol: a
# REF-NEXT:       Addend: -10
# REF-NEXT:     }
# REF-NEXT:     Relocation {
# REF-NEXT:       Type: R_WASM_TYPE_INDEX_LEB (6)
# REF-NEXT:       Offset: 0xD
# REF-NEXT:       Index: 0x1
# REF-NEXT:     }
# REF-NEXT:     Relocation {
# REF-NEXT:       Type: R_WASM_TABLE_NUMBER_LEB (20)
# REF-NEXT:       Offset: 0x12
# REF-NEXT:       Symbol: __indirect_function_table
# REF-NEXT:     }
# REF-NEXT:     Relocation {
# REF-NEXT:       Type: R_WASM_MEMORY_ADDR_LEB (3)
# REF-NEXT:       Offset: 0x1C
# REF-NEXT:       Symbol: b
# REF-NEXT:       Addend: 20
# REF-NEXT:     }
# REF-NEXT:     Relocation {
# REF-NEXT:       Type: R_WASM_TYPE_INDEX_LEB (6)
# REF-NEXT:       Offset: 0x22
# REF-NEXT:       Index: 0x0
# REF-NEXT:     }
# REF-NEXT:     Relocation {
# REF-NEXT:       Type: R_WASM_TABLE_NUMBER_LEB (20)
# REF-NEXT:       Offset: 0x27
# REF-NEXT:       Symbol: __indirect_function_table
# REF-NEXT:     }
# REF-NEXT:     Relocation {
# REF-NEXT:       Type: R_WASM_FUNCTION_INDEX_LEB (0)
# REF-NEXT:       Offset: 0x2E
# REF-NEXT:       Symbol: c
# REF-NEXT:     }
# REF-NEXT:     Relocation {
# REF-NEXT:       Type: R_WASM_FUNCTION_INDEX_LEB (0)
# REF-NEXT:       Offset: 0x35
# REF-NEXT:       Symbol: d
# REF-NEXT:     }
# REF-NEXT:   }
# REF-NEXT: ]
