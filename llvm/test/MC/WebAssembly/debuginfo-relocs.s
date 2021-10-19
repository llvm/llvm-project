# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t.o %s
# RUN: obj2yaml %t.o | FileCheck %s

.functype undef () -> ()

bar:
    .functype bar () -> ()
    end_function

    .globl _start
_start:
    .functype _start () -> ()
    call bar
    end_function

.section .debug_int,"",@
.Ld:
  .int32 1
.size .Ld, 4

.section .debug_info,"",@
    .int32 bar
    .int32 undef
    .int32 .Ld

## Test that relocations in metadata sections against both defined and undef
## function symbols get R_WASM_FUNCTION_OFFSET relocations, and relocs against
## data symbols get R_WASM_SECTION_OFFSET relocs.
# CHECK:     - Type: CUSTOM
# CHECK-NEXT:  Name: .debug_int
# CHECK:     - Type: CUSTOM
# CHECK-NEXT:    Relocations:
# CHECK-NEXT:      - Type:            R_WASM_FUNCTION_OFFSET_I32
# CHECK-NEXT:        Index:           0
# CHECK-NEXT:        Offset:          0x0
# CHECK-NEXT:      - Type:            R_WASM_FUNCTION_OFFSET_I32
# CHECK-NEXT:        Index:           3
# CHECK-NEXT:        Offset:          0x4
# CHECK-NEXT:      - Type:            R_WASM_SECTION_OFFSET_I32
# CHECK-NEXT:        Index:           2
# CHECK-NEXT:        Offset:          0x8
# CHECK-NEXT:         Name:            .debug_info

