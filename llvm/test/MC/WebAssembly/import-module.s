# RUN: llvm-mc -triple=wasm32 < %s | FileCheck %s -check-prefix=CHECK-ASM
# RUN: llvm-mc -triple=wasm32 -filetype=obj -o - < %s | obj2yaml | FileCheck %s

.functype foo () -> ()
.functype plain () -> ()
.functype __wasm_component_model_builtin_context_get_0 () -> (i32)
.functype mid$dollar () -> ()
.functype mid?question () -> ()

test:
  .functype test () -> ()
  call      foo
  call      plain
  call     __wasm_component_model_builtin_context_get_0
  drop
  call     mid$dollar
  call     mid?question
  end_function

  .import_module  foo, bar
  .import_name  foo, qux

  .import_module __wasm_component_model_builtin_context_get_0, "$root"
  .import_name __wasm_component_model_builtin_context_get_0, "[context-get-0]"

  .import_module mid$dollar, another$mid$dollar
  .import_name mid$dollar, mid$dollar

  .import_module mid?question, another?mid?question
  .import_name mid?question, mid?question

# CHECK-ASM: .import_module  foo, bar
# CHECK-ASM: .import_name  foo, qux
# CHECK-ASM: .import_module __wasm_component_model_builtin_context_get_0, "$root"
# CHECK-ASM: .import_name __wasm_component_model_builtin_context_get_0, "[context-get-0]"
# CHECK-ASM: .import_module mid$dollar, another$mid$dollar
# CHECK-ASM: .import_name mid$dollar, mid$dollar
# CHECK-ASM: .import_module mid?question, another?mid?question
# CHECK-ASM: .import_name mid?question, mid?question

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

# CHECK:            - Module:          'another$mid$dollar'
# CHECK-NEXT:         Field:           'mid$dollar'
# CHECK-NEXT:         Kind:            FUNCTION

# CHECK:            - Module:          'another?mid?question'
# CHECK-NEXT:         Field:           'mid?question'
# CHECK-NEXT:         Kind:            FUNCTION

# CHECK:        - Type:            CUSTOM
# CHECK:              Name:            foo
# CHECK-NEXT:         Flags:           [ UNDEFINED, EXPLICIT_NAME ]

# CHECK:              Name:            plain
# CHECK-NEXT:         Flags:           [ UNDEFINED ]

# CHECK:              Name:            __wasm_component_model_builtin_context_get_0
# CHECK-NEXT:         Flags:           [ UNDEFINED, EXPLICIT_NAME ]

# CHECK:              Name:            'mid$dollar'
# CHECK-NEXT:         Flags:           [ UNDEFINED, EXPLICIT_NAME ]

# CHECK:              Name:            'mid?question'
# CHECK-NEXT:         Flags:           [ UNDEFINED, EXPLICIT_NAME ]
