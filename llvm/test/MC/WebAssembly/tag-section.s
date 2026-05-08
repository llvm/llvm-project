# RUN: llvm-mc -triple=wasm32-unknown-unknown -mattr=+exception-handling -filetype=obj %s | obj2yaml | FileCheck %s
# RUN: llvm-mc -triple=wasm32-unknown-unknown -mattr=+exception-handling -filetype=obj %s | llvm-readobj -S - | FileCheck -check-prefix=SEC %s

.tagtype my_exception i32

.globl test_throw0
test_throw0:
  .functype test_throw0 (i32) -> (i32)
  i32.const 0
  throw my_exception
  i32.const 0
  end_function

.globl test_throw1
test_throw1:
  .functype test_throw1 (i32) -> (i32)
  i32.const 0
  throw my_exception
  i32.const 0
  end_function

.globl my_exception
my_exception:

# CHECK:      Sections:
# CHECK-NEXT:   - Type:            TYPE
# CHECK-NEXT:     Signatures:
# CHECK-NEXT:       - Index:           0
# CHECK-NEXT:         ParamTypes:
# CHECK-NEXT:           - I32
# CHECK-NEXT:         ReturnTypes:
# CHECK-NEXT:           - I32
# CHECK-NEXT:       - Index:           1
# CHECK-NEXT:         ParamTypes:
# CHECK-NEXT:           - I32
# CHECK-NEXT:         ReturnTypes:      []

# CHECK:        - Type:            TAG
# CHECK-NEXT:     TagTypes:        [ 1 ]

# CHECK-NEXT:   - Type:            CODE
# CHECK-NEXT:     Relocations:
# CHECK-NEXT:       - Type:            R_WASM_TAG_INDEX_LEB
# CHECK-NEXT:         Index:           1
# CHECK-NEXT:         Offset:          0x6
# CHECK-NEXT:       - Type:            R_WASM_TAG_INDEX_LEB
# CHECK-NEXT:         Index:           1
# CHECK-NEXT:         Offset:          0x13

# CHECK:        - Type:            CUSTOM
# CHECK-NEXT:     Name:            linking
# CHECK-NEXT:     Version:         2
# CHECK-NEXT:     SymbolTable:

# CHECK:            - Index:           1
# CHECK-NEXT:         Kind:            TAG
# CHECK-NEXT:         Name:            my_exception
# CHECK-NEXT:         Flags:           [ ]
# CHECK-NEXT:         Tag:             0

# SEC:          Type: TAG (0xD)
# SEC-NEXT:     Size: 3
# SEC-NEXT:     Offset: 69
