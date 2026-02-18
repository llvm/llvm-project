# RUN: llvm-mc -triple=wasm32 < %s | FileCheck %s -check-prefix=CHECK-ASM
# RUN: llvm-mc -triple=wasm32 -filetype=obj -o - < %s | obj2yaml | FileCheck %s

.functype foo () -> ()
.functype plain () -> ()
.functype __wasm_component_model_builtin_context_get_0 () -> (i32)

test:
  .functype test () -> ()
  call      foo
  call      plain
  call     __wasm_component_model_builtin_context_get_0
  drop
  end_function

  .import_module  foo, bar
  .import_name  foo, qux

  .import_module __wasm_component_model_builtin_context_get_0, "$root"
  .import_name __wasm_component_model_builtin_context_get_0, "[context-get-0]"

# CHECK-ASM: .import_module  foo, bar
# CHECK-ASM: .import_name  foo, qux

# CHECK:        - Type:            IMPORT
# CHECK-NEXT:     Imports:
# CHECK:            - Module:          bar
# CHECK-NEXT:         Field:           qux
# CHECK-NEXT:         Kind:            FUNCTION

# CHECK:            - Module:          env
# CHECK-NEXT:         Field:           plain
# CHECK-NEXT:         Kind:            FUNCTION

# CHECK:            - Module:          '$root'
# CHECK-NEXT:         Field:           '[context-get-0]'
# CHECK-NEXT:         Kind:            FUNCTION
# CHECK-NEXT:         SigIndex:        1

# CHECK:        - Type:            CUSTOM
# CHECK:              Name:            foo
# CHECK-NEXT:         Flags:           [ UNDEFINED, EXPLICIT_NAME ]

# CHECK:              Name:            plain
# CHECK-NEXT:         Flags:           [ UNDEFINED ]

# CHECK:              Name:            __wasm_component_model_builtin_context_get_0
# CHECK-NEXT:         Flags:           [ UNDEFINED, EXPLICIT_NAME ]