# RUN: llvm-mc -triple=wasm32 -filetype=obj %s -o - | obj2yaml | FileCheck %s

        .section .text.foo,"G",@,abc123,comdat
        .globl foo
        .type foo,@function
foo:
        .functype foo () -> ()
        return
        end_function

        .globl bar
bar:
        .functype bar () -> ()
        return
        end_function

        .section .debug_foo,"G",@,abc123,comdat
        .int32 42
        .section .debug_foo,"G",@,duplicate,comdat
        .int64 234

# Check that there are 2 identically-named custom sections, with the desired
# contents
# CHECK:  - Type:            CUSTOM
# CHECK-NEXT:    Name:            .debug_foo
# CHECK-NEXT:    Payload:         2A000000
# CHECK-NEXT:  - Type:            CUSTOM
# CHECK-NEXT:    Name:            .debug_foo
# CHECK-NEXT:    Payload:         EA00000000000000

# And check that they are in 2 different comdat groups
# CHECK-NEXT:- Type:            CUSTOM
# CHECK-NEXT:    Name:            linking
# CHECK-NEXT:    Version:         2
# CHECK:    Comdats:
# CHECK-NEXT:      - Name:            abc123
# CHECK-NEXT:        Entries:
# CHECK-NEXT:          - Kind:            FUNCTION
# CHECK-NEXT:            Index:           0

# If the user forgets to create a new section for a function, one is created for
# them by the assembler. Check that it is also in the same group.
# CHECK-NEXT:          - Kind:            FUNCTION
# CHECK-NEXT:            Index:           1
# CHECK-NEXT:          - Kind:            SECTION
# CHECK-NEXT:            Index:           4
# CHECK-NEXT:      - Name:            duplicate
# CHECK-NEXT:        Entries:
# CHECK-NEXT:          - Kind:            SECTION
# CHECK-NEXT:            Index:           5
