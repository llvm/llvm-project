# RUN: llvm-mc -mattr=+reference-types -triple=wasm32-unknown-unknown -filetype=obj -o - < %s | obj2yaml | FileCheck %s

.globl __indirect_function_table
.tabletype __indirect_function_table, funcref

.globl obtain_funcref_from_table_index

obtain_funcref_from_table_index:
  .functype obtain_funcref_from_table_index (i32) -> (funcref)
  local.get 0
  table.get __indirect_function_table
  end_function

#      CHECK:  Sections:
# CHECK-NEXT:    - Type:            TYPE
# CHECK-NEXT:      Signatures:
# CHECK-NEXT:        - Index:           0
# CHECK-NEXT:          ParamTypes:
# CHECK-NEXT:            - I32
# CHECK-NEXT:          ReturnTypes:
# CHECK-NEXT:            - FUNCREF
