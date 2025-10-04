# RUN: llvm-mc -triple=wasm32 %s | FileCheck --check-prefix=PRINT %s
# RUN: llvm-mc -filetype=obj -triple=wasm32 %s | llvm-readobj -r - | FileCheck %s

get_addr_func:
  .functype get_addr_func () -> (i32)
  i32.const 0
  nop # 4 NOPs in addition to one zero in i32.const 0 for a canonical 5 byte relocatable [S]LEB.
  nop
  nop
  nop
  end_function

# PRINT: .reloc get_addr_func+2, R_WASM_MEMORY_ADDR_SLEB, function_index_data+1
# CHECK:      Section ({{.*}}) CODE {
# CHECK-NEXT:   0x4 R_WASM_MEMORY_ADDR_SLEB function_index_data 1
# CHECK-NEXT: }
.reloc get_addr_func + 2, R_WASM_MEMORY_ADDR_SLEB, function_index_data + 1

.section .data,"",@
function_index_data:
  .int32 0
.size function_index_data, 4

# PRINT: .reloc function_index_data, R_WASM_FUNCTION_INDEX_I32, get_addr_func
# CHECK:      Section ({{.*}}) DATA {
# CHECK-NEXT:   0x6 R_WASM_FUNCTION_INDEX_I32 get_addr_func
# CHECK-NEXT: }
.reloc function_index_data, R_WASM_FUNCTION_INDEX_I32, get_addr_func
