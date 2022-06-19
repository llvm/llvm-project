# RUN: llvm-mc -triple=wasm64-unknown-unknown < %s | FileCheck %s
# RUN: llvm-mc -triple=wasm64-unknown-unknown -filetype=obj -o %t.o < %s
# RUN: obj2yaml %t.o | FileCheck %s --check-prefix=CHECK-OBJ --match-full-lines

.globaltype __tls_base, i64

tls_store:
  .functype tls_store (i32) -> ()
  # CHECK: global.get __tls_base
  # CHECK-NEXT: i64.const tls1@TLSREL
  # CHECK-NEXT: i64.add
  # CHECK-NEXT: local.get 0
  # CHECK-NEXT: i32.store 0
  global.get __tls_base
  i64.const tls1@TLSREL
  i64.add
  local.get 0
  i32.store 0
  end_function

.section .tls.foo,"T",@
# CHECK: .tls.foo,"T",@
tls1:
  .int32 42
  .size tls1, 4

.section custom_tls,"T",@
# CHECK: custom_tls,"T",@
tls2:
  .int32 43
  .size tls2, 4

#      CHECK-OBJ:  - Type:            CODE
# CHECK-OBJ-NEXT:    Relocations:
# CHECK-OBJ-NEXT:      - Type:            R_WASM_GLOBAL_INDEX_LEB
# CHECK-OBJ-NEXT:        Index:           1
# CHECK-OBJ-NEXT:        Offset:          0x4
# CHECK-OBJ-NEXT:      - Type:            R_WASM_MEMORY_ADDR_TLS_SLEB64
# CHECK-OBJ-NEXT:        Index:           2
# CHECK-OBJ-NEXT:        Offset:          0xA

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
# CHECK-OBJ-NEXT:        Kind:            GLOBAL
# CHECK-OBJ-NEXT:        Name:            __tls_base
# CHECK-OBJ-NEXT:        Flags:           [ UNDEFINED ]
# CHECK-OBJ-NEXT:        Global:          0
# CHECK-OBJ-NEXT:      - Index:           2
# CHECK-OBJ-NEXT:        Kind:            DATA
# CHECK-OBJ-NEXT:        Name:            tls1
# CHECK-OBJ-NEXT:        Flags:           [ BINDING_LOCAL, TLS ]
# CHECK-OBJ-NEXT:        Segment:         0
# CHECK-OBJ-NEXT:        Size:            4
# CHECK-OBJ-NEXT:      - Index:           3
# CHECK-OBJ-NEXT:        Kind:            DATA
# CHECK-OBJ-NEXT:        Name:            tls2
# CHECK-OBJ-NEXT:        Flags:           [ BINDING_LOCAL, TLS ]
# CHECK-OBJ-NEXT:        Segment:         1
# CHECK-OBJ-NEXT:        Size:            4
# CHECK-OBJ-NEXT:    SegmentInfo:
# CHECK-OBJ-NEXT:      - Index:           0
# CHECK-OBJ-NEXT:        Name:            .tls.foo
# CHECK-OBJ-NEXT:        Alignment:       0
# CHECK-OBJ-NEXT:        Flags:           [ TLS ]
# CHECK-OBJ-NEXT:      - Index:           1
# CHECK-OBJ-NEXT:        Name:            custom_tls
# CHECK-OBJ-NEXT:        Alignment:       0
# CHECK-OBJ-NEXT:        Flags:           [ TLS ]
