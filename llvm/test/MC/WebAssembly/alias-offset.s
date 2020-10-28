# RUN: llvm-mc -triple=wasm32-unknown-unknown -filetype=obj %s | llvm-objdump --triple=wasm32-unknown-unknown -d -t -r - | FileCheck %s

  .section    .data,"",@
foo:
  .int32 0
  .size foo, 4
sym_a:
  .int32 1
  .int32 2
  .size sym_a, 8

.set sym_b, sym_a + 4

# CHECK-LABEL: SYMBOL TABLE:
# CHECK-NEXT: 00000000 l     O DATA foo
# CHECK-NEXT: 00000004 l     O DATA sym_a
# CHECK-NEXT: 00000008 l     O DATA sym_b
# CHECK-NEXT: 00000001 l     F CODE main

  .text
  .section    .text,"",@
main:
  .functype   main () -> ()
  i32.const sym_a
  i32.store sym_b
  end_function

# CHECK-LABEL: <main>:
# CHECK-EMPTY:
# CHECK-NEXT:       3: 41 88 80 80 80 00     i32.const       8
# CHECK-NEXT:                        00000004:  R_WASM_MEMORY_ADDR_SLEB      sym_a+0
# CHECK-NEXT:       9: 36 02 8c 80 80 80 00  i32.store       12
# CHECK-NEXT:                        0000000b:  R_WASM_MEMORY_ADDR_LEB      sym_b+0
# CHECK-NEXT:      10: 0b            end
