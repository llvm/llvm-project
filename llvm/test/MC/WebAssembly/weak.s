# RUN: llvm-mc -triple=wasm32-unknown-unknown -filetype=obj -o %t.o < %s
# RUN: obj2yaml %t.o | FileCheck %s

weak_function:
  .functype weak_function () -> (i32)
  .hidden weak_function
  .weak weak_function
  i32.const 0
  i32.load weak_external_data
  end_function

.weak weak_external_data

# CHECK:          SymbolTable:
# CHECK-NEXT:       - Index:           0
# CHECK-NEXT:         Kind:            FUNCTION
# CHECK-NEXT:         Name:            weak_function
# CHECK-NEXT:         Flags:           [ BINDING_WEAK, VISIBILITY_HIDDEN ]
# CHECK-NEXT:         Function:        0
# CHECK-NEXT:       - Index:           1
# CHECK-NEXT:         Kind:            DATA
# CHECK-NEXT:         Name:            weak_external_data
# CHECK-NEXT:         Flags:           [ BINDING_WEAK, UNDEFINED ]
# CHECK-NEXT: ...
