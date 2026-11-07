//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "llvm/Option/LibraryOptions.td"

def SelectionDAGOptions : OptionsStruct;

// DAGCombiner.cpp
defm : BoolField<"combiner-use-tbaa", "true",
  "Enable DAG combiner's use of TBAA">;

defm : BoolField<"combiner-stress-load-slicing", "false",
  "Bypass the profitability model of load slicing">;

defm : BoolField<"combiner-split-load-index", "true",
  "DAG combiner may split indexing from loads">;
