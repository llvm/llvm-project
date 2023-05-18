# RUN: llvm-mc -triple=wasm32 < %s | FileCheck %s -check-prefix=CHECK-ASM
# RUN: llvm-mc -triple=wasm32 -filetype=obj -o - < %s | obj2yaml | FileCheck %s

.functype foo () -> ()
.functype plain () -> ()

test:
  .functype test () -> ()
  call      foo
  call      plain
  end_function

  .import_module  foo, bar
  .import_name  foo, qux

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

# CHECK:        - Type:            CUSTOM
# CHECK:              Name:            foo
# CHECK-NEXT:         Flags:           [ UNDEFINED, EXPLICIT_NAME ]

# CHECK:              Name:            plain
# CHECK-NEXT:         Flags:           [ UNDEFINED ]
