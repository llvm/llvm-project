# RUN: llvm-mc -triple=wasm32 < %s | FileCheck %s -check-prefix=CHECK-ASM
# RUN: llvm-mc -triple=wasm32 -filetype=obj -o - < %s | obj2yaml | FileCheck %s

.functype foo () -> ()
.functype plain () -> ()
.functype utf8_import() -> ()
.functype mid$dollar () -> ()
.functype mid?question () -> ()

test:
  .functype test () -> ()
  call      foo
  call      plain
  call      utf8_import
  call      mid$dollar
  call      mid?question
  end_function

  .import_module  foo, bar
  .import_name  foo, qux

  .import_module utf8_import, "$café-Straßburg"
  .import_name utf8_import, "[café-Straßburg]"

  .import_module mid$dollar, another$mid$dollar
  .import_name mid$dollar, mid$dollar$name

  .import_module mid?question, another?mid?question
  .import_name mid?question, mid?question?name

# CHECK-ASM: .import_module  foo, "bar"
# CHECK-ASM: .import_name  foo, "qux"
# CHECK-ASM: .import_module utf8_import, "$café-Straßburg"
# CHECK-ASM: .import_name utf8_import, "[café-Straßburg]"
# CHECK-ASM: .import_module mid$dollar, "another$mid$dollar"
# CHECK-ASM: .import_name mid$dollar, "mid$dollar$name"
# CHECK-ASM: .import_module mid?question, "another?mid?question"
# CHECK-ASM: .import_name mid?question, "mid?question?name"

# CHECK:        - Type:            IMPORT
# CHECK-NEXT:     Imports:
# CHECK:            - Module:          bar
# CHECK-NEXT:         Field:           qux
# CHECK-NEXT:         Kind:            FUNCTION

# CHECK:            - Module:          env
# CHECK-NEXT:         Field:           plain
# CHECK-NEXT:         Kind:            FUNCTION

# CHECK:            - Module:          "$café-Straßburg"
# CHECK-NEXT:         Field:           "[café-Straßburg]"
# CHECK-NEXT:         Kind:            FUNCTION

# CHECK:            - Module:          'another$mid$dollar'
# CHECK-NEXT:         Field:           'mid$dollar$name'
# CHECK-NEXT:         Kind:            FUNCTION

# CHECK:            - Module:          'another?mid?question'
# CHECK-NEXT:         Field:           'mid?question?name'
# CHECK-NEXT:         Kind:            FUNCTION

# CHECK:        - Type:            CUSTOM
# CHECK:              Name:            foo
# CHECK-NEXT:         Flags:           [ UNDEFINED, EXPLICIT_NAME ]

# CHECK:              Name:            plain
# CHECK-NEXT:         Flags:           [ UNDEFINED ]

# CHECK:              Name:            utf8_import
# CHECK-NEXT:         Flags:           [ UNDEFINED, EXPLICIT_NAME ]

# CHECK:              Name:            'mid$dollar'
# CHECK-NEXT:         Flags:           [ UNDEFINED, EXPLICIT_NAME ]

# CHECK:              Name:            'mid?question'
# CHECK-NEXT:         Flags:           [ UNDEFINED, EXPLICIT_NAME ]
