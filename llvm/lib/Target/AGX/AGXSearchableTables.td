//===-- AGXSearchableTables.td - AGX lookup tables ---------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "llvm/TableGen/SearchableTable.td"

// Pseudo instruction to real instruction, per encoding family.
def AGXPseudoToRealTable : GenericTable {
  let FilterClass = "AGXPseudo";
  let CppTypeName = "AGXPseudoToReal";
  let Fields = ["Pseudo", "G13Real", "Apple9Real"];
  let PrimaryKey = ["Pseudo"];
  let PrimaryKeyName = "getAGXPseudoToReal";
}
