# RUN: llvm-mc -triple=wasm32 %s | FileCheck --check-prefix=PRINT %s
# RUN: llvm-mc -filetype=obj -triple=wasm32 %s | llvm-readobj -r - | FileCheck %s

load_function_index_func:
  .functype load_function_index_func () -> (i32)
  i32.const 0
  nop
  nop
  nop
  nop
  i32.load  0
  end_function

# PRINT: .reloc load_function_index_func+2, R_WASM_MEMORY_ADDR_SLEB, function_index_data+1
# CHECK:      Section ({{.*}}) CODE {
# CHECK-NEXT:   0x4 R_WASM_MEMORY_ADDR_SLEB function_index_data 1
# CHECK-NEXT: }
.reloc load_function_index_func + 2, R_WASM_MEMORY_ADDR_SLEB, function_index_data + 1

.section .data,"",@
function_index_data:
  .int32 0
.size function_index_data, 4

# PRINT: .reloc function_index_data, R_WASM_FUNCTION_INDEX_I32, load_function_index_func
# CHECK:      Section ({{.*}}) DATA {
# CHECK-NEXT:   0x6 R_WASM_FUNCTION_INDEX_I32 load_function_index_func
# CHECK-NEXT: }
.reloc function_index_data, R_WASM_FUNCTION_INDEX_I32, load_function_index_func
