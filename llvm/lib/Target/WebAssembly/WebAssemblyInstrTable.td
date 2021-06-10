// WebAssemblyInstrTable.td - WebAssembly Table codegen support -*- tablegen -*-
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// WebAssembly Table operand code-gen constructs.
/// Instructions that handle tables
//===----------------------------------------------------------------------===//

multiclass TABLE<WebAssemblyRegClass rt> {
  let mayLoad = 1 in
  defm TABLE_GET_#rt : I<(outs rt:$res), (ins table32_op:$table, I32:$i),
                         (outs), (ins table32_op:$table),
                         [],
                         "table.get\t$res, $table, $i",
                         "table.get\t$table",
                         0x25>;

  let mayStore = 1 in
  defm TABLE_SET_#rt : I<(outs), (ins table32_op:$table, I32:$i, rt:$val),
                         (outs), (ins table32_op:$table),
                         [],
                         "table.set\t$table, $i, $val",
                         "table.set\t$table",
                         0x26>;

  defm TABLE_GROW_#rt : I<(outs I32:$sz), (ins table32_op:$table, rt:$val, I32:$n),
                          (outs), (ins table32_op:$table),
                          [],
                          "table.grow\t$sz, $table, $val, $n",
                          "table.grow\t$table",
                          0xfc0f>;

  defm TABLE_FILL_#rt : I<(outs), (ins table32_op:$table, I32:$i, rt:$val, I32:$n),
                          (outs), (ins table32_op:$table),
                          [],
                          "table.fill\t$table, $i, $val, $n",
                          "table.fill\t$table",
                          0xfc11>;

}

defm "" : TABLE<FUNCREF>, Requires<[HasReferenceTypes]>;
defm "" : TABLE<EXTERNREF>, Requires<[HasReferenceTypes]>;

def wasm_table_set_t : SDTypeProfile<0, 3, []>;
def wasm_table_set : SDNode<"WebAssemblyISD::TABLE_SET", wasm_table_set_t,
                            [SDNPHasChain, SDNPMayStore, SDNPMemOperand]>;

def : Pat<(wasm_table_set i32:$table, i32:$idx, funcref:$r),
          (TABLE_SET_FUNCREF i32:$table, i32:$idx, funcref:$r)>,
          Requires<[HasReferenceTypes]>;
def : Pat<(wasm_table_set i32:$table, i32:$idx, externref:$r),
          (TABLE_SET_EXTERNREF i32:$table, i32:$idx, externref:$r)>,
          Requires<[HasReferenceTypes]>;

defm TABLE_SIZE : I<(outs I32:$sz), (ins table32_op:$table),
                    (outs), (ins table32_op:$table),
                    [],
                    "table.size\t$sz, $table",
                    "table.size\t$table",
                    0xfc10>,
                    Requires<[HasReferenceTypes]>;


defm TABLE_COPY : I<(outs), (ins table32_op:$table1, table32_op:$table2, I32:$d, I32:$s, I32:$n),
                    (outs), (ins table32_op:$table1, table32_op:$table2),
                    [],
                    "table.copy\t$table1, $table2, $d, $s, $n",
                    "table.copy\t$table1, $table2",
                    0xfc0e>,
                    Requires<[HasReferenceTypes]>;
