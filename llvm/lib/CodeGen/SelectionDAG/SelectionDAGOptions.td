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
defm : OptionalBoolField<"combiner-global-alias-analysis",
  "Enable DAG combiner's use of IR alias analysis">;
defm : BoolField<"combiner-use-tbaa", "true",
  "Enable DAG combiner's use of TBAA">;
let Condition = AssertsOnly in {
  defm : ValueField<"combiner-aa-only-func", "std::optional<StringRef>", "std::nullopt",
    "Only use DAG-combiner alias analysis in this function">;
}
defm : BoolField<"combiner-stress-load-slicing", "false",
  "Bypass the profitability model of load slicing">;
defm : BoolField<"combiner-split-load-index", "true",
  "DAG combiner may split indexing from loads">;
defm : BoolField<"combiner-store-merging", "true",
  "DAG combiner enable merging multiple stores into a wider store">;
defm : ValueField<"combiner-tokenfactor-inline-limit", "unsigned", "2048",
  "Limit the number of operands to inline for Token Factors">;
defm : ValueField<"combiner-store-merge-dependence-limit", "unsigned", "10",
  "Limit the number of times for the same StoreNode and RootNode "
  "to bail out in store merging dependence check">;
defm : BoolField<"combiner-reduce-load-op-store-width", "true",
  "DAG combiner enable reducing the width of load/op/store sequence">;
defm : BoolField<"combiner-reduce-load-op-store-width-force-narrowing-profitable", "false",
  "DAG combiner force override the narrowing profitable check when "
  "reducing the width of load/op/store sequences">;
defm : BoolField<"combiner-shrink-load-replace-store-with-store", "true",
  "DAG combiner enable load/<replace bytes>/store with a narrower store">;
defm : OptionalBoolField<"combiner-topological-sorting",
  "DAG combiner nodes consistently processed in topological order">;
defm : BoolField<"combiner-disabled", "false",
  "Disable the DAG combiner">;

// LegalizeTypes.cpp
defm : BoolField<"enable-legalize-types-checking", "false",
  "Do extensive, expensive, basic correctness checking in LegalizeTypes">;
