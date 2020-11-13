# RUN: llvm-mc -triple=wasm32-unknown-unknown < %s | FileCheck %s
# RUN: llvm-mc -triple=wasm32-unknown-unknown -filetype=obj -o %t.o < %s
# RUN: obj2yaml %t.o | FileCheck %s --check-prefix=CHECK-OBJ

tls_store:
  .functype tls_store (i32) -> ()
  # CHECK: global.get __tls_base
  # CHECK-NEXT: i32.const tls@TLSREL
  # CHECK-NEXT: i32.add
  # CHECK-NEXT: i32.store 0
  global.get __tls_base
  i32.const tls@TLSREL
  i32.add
  i32.store 0
  end_function


#      CHECK-OBJ:  - Type:            CODE
# CHECK-OBJ-NEXT:    Relocations:
# CHECK-OBJ-NEXT:      - Type:            R_WASM_MEMORY_ADDR_LEB
# CHECK-OBJ-NEXT:        Index:           1
# CHECK-OBJ-NEXT:        Offset:          0x00000004
# CHECK-OBJ-NEXT:      - Type:            R_WASM_MEMORY_ADDR_TLS_SLEB
# CHECK-OBJ-NEXT:        Index:           2
# CHECK-OBJ-NEXT:        Offset:          0x0000000A

#      CHECK-OBJ:  - Type:            CUSTOM
# CHECK-OBJ-NEXT:    Name:            linking
# CHECK-OBJ-NEXT:    Version:         2
# CHECK-OBJ-NEXT:    SymbolTable:
# CHECK-OBJ-NEXT:      - Index:           0
# CHECK-OBJ-NEXT:        Kind:            FUNCTION
# CHECK-OBJ-NEXT:        Name:            tls_store
# CHECK-OBJ-NEXT:        Flags:           [ BINDING_LOCAL ]
# CHECK-OBJ-NEXT:        Function:        0
# CHECK-OBJ-NEXT:      - Index:           1
# CHECK-OBJ-NEXT:        Kind:            DATA
# CHECK-OBJ-NEXT:        Name:            __tls_base
# CHECK-OBJ-NEXT:        Flags:           [ UNDEFINED ]
# CHECK-OBJ-NEXT:      - Index:           2
# CHECK-OBJ-NEXT:        Kind:            DATA
# CHECK-OBJ-NEXT:        Name:            tls
# CHECK-OBJ-NEXT:        Flags:           [ UNDEFINED ]
