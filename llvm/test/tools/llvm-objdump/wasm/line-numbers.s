# REQUIRES: webassembly-registered-target
# RUN: llvm-mc -triple=wasm32-unknown-unknown -filetype=obj %s -o %t.o -g
# RUN: llvm-objdump -d --line-numbers %t.o | FileCheck --check-prefix=OBJ %s

# line-numbers.yaml was created by linking this object and converting to YAML:
#  wasm-ld %t.o -o %t.wasm --no-entry --export=foo --export=bar
# RUN: yaml2obj %S/Inputs/line-numbers.yaml -o %t.wasm
# RUN: llvm-objdump -d --line-numbers %t.wasm | FileCheck --check-prefix=LINKED %s

# This test mirrors test/tools/llvm-symbolizer/wasm-basic.s and tests that line
# numbers are correctly printed from DWARF information.

.globl foo
foo:
    .functype foo () -> ()
    nop
    return
    end_function

.globl bar
bar:
    .functype bar (i32) -> (i32)
    local.get 0
    nop
    return
    end_function

# OBJ:      <foo>:
# OBJ-EMPTY:
# OBJ-NEXT: ; foo():
# OBJ-NEXT: ; {{.*}}line-numbers.s:[[#@LINE-15]]
# OBJ-NEXT:        3: 01            nop
# OBJ-NEXT: ; {{.*}}line-numbers.s:[[#@LINE-16]]
# OBJ-NEXT:        4: 0f            return
# OBJ-NEXT: ; {{.*}}line-numbers.s:[[#@LINE-17]]
# OBJ-NEXT:        5: 0b            end

# OBJ:      <bar>:
# OBJ-EMPTY:
# OBJ-NEXT: ; bar():
# OBJ-NEXT: ; {{.*}}line-numbers.s:[[#@LINE-18]]
# OBJ-NEXT:        8: 20 00         local.get 0
# OBJ-NEXT: ; {{.*}}line-numbers.s:[[#@LINE-19]]
# OBJ-NEXT:        a: 01            nop
# OBJ-NEXT: ; {{.*}}line-numbers.s:[[#@LINE-20]]
# OBJ-NEXT:        b: 0f            return
# OBJ-NEXT: ; {{.*}}line-numbers.s:[[#@LINE-21]]
# OBJ-NEXT:        c: 0b            end


# Note: The linked version of this test currently bakes in the absolute binary
# offsets in the code section (for the beginning of each function) because they
# are relevant. We want to make sure that file offsets are used rather than
# section offsets. But this does mean that if the output of the linker for the
# sections that come before Code changes, then this test will have to be updated.

# LINKED:      <foo>:
# LINKED-EMPTY:
# LINKED-NEXT: ; foo():
# LINKED-NEXT: ; {{.*}}line-numbers.s:[[#@LINE-44]]
# LINKED-NEXT:        5c: 01            nop
# LINKED-NEXT: ; {{.*}}line-numbers.s:[[#@LINE-45]]
# LINKED-NEXT:        {{.*}}: 0f            return
# LINKED-NEXT: ; {{.*}}line-numbers.s:[[#@LINE-46]]
# LINKED-NEXT:        {{.*}}: 0b            end

# LINKED:      <bar>:
# LINKED-EMPTY:
# LINKED-NEXT: ; bar():
# LINKED-NEXT: ; {{.*}}line-numbers.s:[[#@LINE-47]]
# LINKED-NEXT:        61: 20 00         local.get 0
# LINKED-NEXT: ; {{.*}}line-numbers.s:[[#@LINE-48]]
# LINKED-NEXT:        {{.*}}: 01            nop
# LINKED-NEXT: ; {{.*}}line-numbers.s:[[#@LINE-49]]
# LINKED-NEXT:        {{.*}}: 0f            return
# LINKED-NEXT: ; {{.*}}line-numbers.s:[[#@LINE-50]]
# LINKED-NEXT:        {{.*}}: 0b            end
