// RUN: mlir-tblgen -gen-op-decls -I %S/../../include %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=WARN --implicit-check-not=warning: --implicit-check-not=note:
// RUN: mlir-tblgen -gen-op-decls -on-deprecated=none -I %S/../../include %s -o /dev/null 2>&1 | count 0
// RUN: not mlir-tblgen -gen-op-decls -on-deprecated=error -I %S/../../include %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERROR --implicit-check-not=warning: --implicit-check-not=note: --implicit-check-not=error
// RUN: mlir-tblgen -gen-op-decls -op-include-regex="^(results|single)[.]" -I %S/../../include %s -o /dev/null 2>&1 | count 0
// RUN: mlir-tblgen -gen-op-decls -op-include-regex="legacy.single_result" -I %S/../../include %s -o /dev/null 2>&1 | count 0
// RUN: mlir-tblgen -gen-op-defs -I %S/../../include %s -o /dev/null 2>&1 | count 0

// Check that ODS warns once for each dialect that keeps `useOpFoldResults` at
// 0 and has an op with a folder that does not have exactly one fixed result.
// `-on-deprecated=none` silences the warning and `-on-deprecated=error` makes
// it an error.

include "mlir/IR/OpBase.td"

// WARN: :[[# @LINE + 2]]:{{[0-9]+}}: warning: dialect 'legacy' uses the deprecated multi-result fold form; set 'useOpFoldResults' to 1 and return 'OpFoldResults' from each multi-result 'fold'
// ERROR: :[[# @LINE + 1]]:{{[0-9]+}}: error: dialect 'legacy' uses the deprecated multi-result fold form
def Legacy_Dialect : Dialect {
  let name = "legacy";
  let cppNamespace = "legacy";
}
class Legacy_Op<string mnemonic, list<Trait> traits = []> :
    Op<Legacy_Dialect, mnemonic, traits>;

// WARN: :[[# @LINE + 2]]:{{[0-9]+}}: note: 'legacy.optional_result' declares 'LogicalResult fold(FoldAdaptor, SmallVectorImpl<OpFoldResult> &)'
// ERROR: :[[# @LINE + 1]]:{{[0-9]+}}: note: 'legacy.optional_result' declares
def Legacy_OptionalResultOp : Legacy_Op<"optional_result"> {
  let results = (outs Optional<AnyType>:$res);
  let hasFolder = 1;
}

def Legacy_SingleResultOp : Legacy_Op<"single_result"> {
  let results = (outs AnyType:$res);
  let hasFolder = 1;
}

def Legacy_TwoResultsOp : Legacy_Op<"two_results"> {
  let results = (outs AnyType:$lhs, AnyType:$rhs);
  let hasFolder = 1;
}

def Legacy_VariadicResultOp : Legacy_Op<"variadic_result"> {
  let results = (outs Variadic<AnyType>:$res);
  let hasFolder = 1;
}

def Legacy_ZeroResultsOp : Legacy_Op<"zero_results"> {
  let hasFolder = 1;
}

// WARN: :[[# @LINE + 2]]:{{[0-9]+}}: warning: dialect 'other_legacy' uses the deprecated multi-result fold form
// ERROR: :[[# @LINE + 1]]:{{[0-9]+}}: error: dialect 'other_legacy' uses the deprecated multi-result fold form
def OtherLegacy_Dialect : Dialect {
  let name = "other_legacy";
  let cppNamespace = "other_legacy";
}

// WARN: :[[# @LINE + 2]]:{{[0-9]+}}: note: 'other_legacy.zero_results' declares
// ERROR: :[[# @LINE + 1]]:{{[0-9]+}}: note: 'other_legacy.zero_results' declares
def OtherLegacy_ZeroResultsOp : Op<OtherLegacy_Dialect, "zero_results"> {
  let hasFolder = 1;
}

// A dialect with the bit does not warn.
def Results_Dialect : Dialect {
  let name = "results";
  let cppNamespace = "results";
  let useOpFoldResults = 1;
}

def Results_TwoResultsOp : Op<Results_Dialect, "two_results"> {
  let results = (outs AnyType:$lhs, AnyType:$rhs);
  let hasFolder = 1;
}

// A dialect without the bit does not warn when it has no multi-result folder.
def Single_Dialect : Dialect {
  let name = "single";
  let cppNamespace = "single";
}

def Single_SingleResultOp : Op<Single_Dialect, "single_result"> {
  let results = (outs AnyType:$res);
  let hasFolder = 1;
}

def Single_TwoResultsNoFolderOp : Op<Single_Dialect, "two_results_no_folder"> {
  let results = (outs AnyType:$lhs, AnyType:$rhs);
}
