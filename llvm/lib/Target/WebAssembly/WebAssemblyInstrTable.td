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
  defm TABLE_GET_#rt : I<(outs rt:$res), (ins table32_op:$table),
                         (outs), (ins table32_op:$table),
                         [],
                         "table.get\t$res, $table",
                         "table.get\t$table",
                         0x25>;

  defm TABLE_SET_#rt : I<(outs), (ins table32_op:$table, rt:$val, I32:$i),
                         (outs), (ins table32_op:$table),
                         [],
                         "table.set\t$table, $val, $i",
                         "table.set\t$table",
                         0x26>;

  defm TABLE_GROW_#rt : I<(outs I32:$sz), (ins table32_op:$table, I32:$n, rt:$val),
                          (outs), (ins table32_op:$table),
                          [],
                          "table.grow\t$sz, $table, $n, $val",
                          "table.grow\t$table",
                          0xfc0f>;

  defm TABLE_FILL_#rt : I<(outs), (ins table32_op:$table, I32:$n, rt:$val, I32:$i),
                          (outs), (ins table32_op:$table),
                          [],
                          "table.fill\t$table, $n, $val, $i",
                          "table.fill\t$table",
                          0xfc11>;

}

defm "" : TABLE<FUNCREF>, Requires<[HasReferenceTypes]>;
defm "" : TABLE<EXTERNREF>, Requires<[HasReferenceTypes]>;

defm TABLE_SIZE : I<(outs I32:$sz), (ins table32_op:$table),
                    (outs), (ins table32_op:$table),
                    [],
                    "table.size\t$sz, $table",
                    "table.size\t$table",
                    0xfc10>,
                    Requires<[HasReferenceTypes]>;


defm TABLE_COPY : I<(outs), (ins table32_op:$table1, table32_op:$table2, I32:$n, I32:$s, I32:$d),
                    (outs), (ins table32_op:$table1, table32_op:$table2),
                    [],
                    "table.copy\t$table1, $table2, $n, $s, $d",
                    "table.copy\t$table1, $table2",
                    0xfc0e>,
                    Requires<[HasReferenceTypes]>;
