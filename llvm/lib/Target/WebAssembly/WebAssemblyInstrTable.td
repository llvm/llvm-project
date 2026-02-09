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

def WebAssemblyTableSet_t : SDTypeProfile<0, 3, [SDTCisPtrTy<1>]>;
def WebAssemblyTableSet : SDNode<"WebAssemblyISD::TABLE_SET", WebAssemblyTableSet_t,
                                 [SDNPHasChain, SDNPMayStore, SDNPMemOperand]>;

def WebAssemblyTableGet_t : SDTypeProfile<1, 2, [SDTCisPtrTy<1>]>;
def WebAssemblyTableGet : SDNode<"WebAssemblyISD::TABLE_GET", WebAssemblyTableGet_t,
                                 [SDNPHasChain, SDNPMayLoad, SDNPMemOperand]>;


multiclass TABLE<WebAssemblyRegClass rc, string suffix, WebAssemblyRegClass index_rc, ValueType index_vt, string addrmodesuffix> {
  let mayLoad = 1 in
  defm TABLE_GET_#rc#addrmodesuffix : I<(outs rc:$res), (ins table32_op:$table, index_rc:$i),
                         (outs), (ins table32_op:$table),
                         [(set rc:$res, (!cast<Intrinsic>("int_wasm_table_get_" # suffix) (WebAssemblyWrapper tglobaladdr:$table), index_rc:$i))],
                         "table.get\t$res, $table, $i",
                         "table.get\t$table",
                         0x25>;

  let mayStore = 1 in
  defm TABLE_SET_#rc#addrmodesuffix : I<(outs), (ins table32_op:$table, index_rc:$i, rc:$val),
                         (outs), (ins table32_op:$table),
                         [(!cast<Intrinsic>("int_wasm_table_set_" # suffix) (WebAssemblyWrapper tglobaladdr:$table), index_rc:$i, rc:$val)],
                         "table.set\t$table, $i, $val",
                         "table.set\t$table",
                         0x26>;

  defm TABLE_GROW_#rc#addrmodesuffix : I<(outs I32:$sz), (ins table32_op:$table, rc:$val, index_rc:$n),
                          (outs), (ins table32_op:$table),
                          [(set I32:$sz, (!cast<Intrinsic>("int_wasm_table_grow_" # suffix) (WebAssemblyWrapper tglobaladdr:$table), rc:$val, index_rc:$n))],
                          "table.grow\t$sz, $table, $val, $n",
                          "table.grow\t$table",
                          0xfc0f>;

  defm TABLE_FILL_#rc#addrmodesuffix : I<(outs), (ins table32_op:$table, index_rc:$i, rc:$val, index_rc:$n),
                          (outs), (ins table32_op:$table),
                          [(!cast<Intrinsic>("int_wasm_table_fill_" # suffix) (WebAssemblyWrapper tglobaladdr:$table), index_rc:$i, rc:$val, index_rc:$n)],
                          "table.fill\t$table, $i, $val, $n",
                          "table.fill\t$table",
                          0xfc11>;

  foreach vt = rc.RegTypes in {
    def : Pat<(vt (WebAssemblyTableGet (WebAssemblyWrapper tglobaladdr:$table), index_vt:$idx)),
              (!cast<NI>("TABLE_GET_" # rc # addrmodesuffix) tglobaladdr:$table, index_vt:$idx)>;
    def : Pat<(WebAssemblyTableSet
               (WebAssemblyWrapper tglobaladdr:$table),
               index_vt:$idx,
               vt:$src),
              (!cast<NI>("TABLE_SET_" # rc # addrmodesuffix) tglobaladdr:$table, index_vt:$idx, vt:$src)>;
  }
}

defm "" : TABLE<FUNCREF, "funcref", I32, i32, "_A32">, Requires<[HasReferenceTypes, HasAddr32]>;
defm "" : TABLE<EXTERNREF, "externref", I32, i32, "_A32">, Requires<[HasReferenceTypes, HasAddr32]>;
defm "" : TABLE<EXNREF, "exnref", I32, i32, "_A32">,
          Requires<[HasReferenceTypes, HasExceptionHandling, HasAddr32]>;

defm "" : TABLE<FUNCREF, "funcref", I64, i64, "_A64">, Requires<[HasReferenceTypes, HasAddr64]>;
defm "" : TABLE<EXTERNREF, "externref", I64, i64, "_A64">, Requires<[HasReferenceTypes, HasAddr64]>;
defm "" : TABLE<EXNREF, "exnref", I64, i64, "_A64">,
          Requires<[HasReferenceTypes, HasExceptionHandling, HasAddr64]>;

def : Pat<(WebAssemblyTableSet mcsym:$table, i32:$idx, funcref:$r),
          (TABLE_SET_FUNCREF_A32 mcsym:$table, i32:$idx, funcref:$r)>,
          Requires<[HasReferenceTypes, HasAddr32]>;

def : Pat<(WebAssemblyTableSet mcsym:$table, i64:$idx, funcref:$r),
          (TABLE_SET_FUNCREF_A64 mcsym:$table, i64:$idx, funcref:$r)>,
          Requires<[HasReferenceTypes, HasAddr64]>;

defm TABLE_SIZE_A32 : I<(outs I32:$sz), (ins table32_op:$table),
                    (outs), (ins table32_op:$table),
                    [(set I32:$sz, (int_wasm_table_size (WebAssemblyWrapper tglobaladdr:$table)))],
                    "table.size\t$sz, $table",
                    "table.size\t$table",
                    0xfc10>,
                    Requires<[HasReferenceTypes, HasAddr32]>;

defm TABLE_SIZE_A64 : I<(outs I64:$sz), (ins table32_op:$table),
                    (outs), (ins table32_op:$table),
                    [(set I64:$sz, (int_wasm_table_size (WebAssemblyWrapper tglobaladdr:$table)))],
                    "table.size\t$sz, $table",
                    "table.size\t$table",
                    0xfc10>,
                    Requires<[HasReferenceTypes, HasAddr64]>;

defm TABLE_COPY_A32 : I<(outs), (ins table32_op:$table1, table32_op:$table2, I32:$d, I32:$s, I32:$n),
                    (outs), (ins table32_op:$table1, table32_op:$table2),
                    [(int_wasm_table_copy (WebAssemblyWrapper tglobaladdr:$table1),
                                          (WebAssemblyWrapper tglobaladdr:$table2),
                                          I32:$d, I32:$s, I32:$n)],
                    "table.copy\t$table1, $table2, $d, $s, $n",
                    "table.copy\t$table1, $table2",
                    0xfc0e>,
                    Requires<[HasReferenceTypes, HasAddr32]>;

defm TABLE_COPY_A64 : I<(outs), (ins table32_op:$table1, table32_op:$table2, I64:$d, I64:$s, I64:$n),
                    (outs), (ins table32_op:$table1, table32_op:$table2),
                    [(int_wasm_table_copy (WebAssemblyWrapper tglobaladdr:$table1),
                                          (WebAssemblyWrapper tglobaladdr:$table2),
                                          I64:$d, I64:$s, I64:$n)],
                    "table.copy\t$table1, $table2, $d, $s, $n",
                    "table.copy\t$table1, $table2",
                    0xfc0e>,
                    Requires<[HasReferenceTypes, HasAddr64]>;
