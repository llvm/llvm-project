// RUN: mlir-tblgen -gen-op-decls -I %S/../../include %s | FileCheck %s

// Check the `fold` declaration that ODS emits with and without the dialect bit
// `useOpFoldResults`.

include "mlir/IR/OpBase.td"

def Legacy_Dialect : Dialect {
  let name = "legacy";
  let cppNamespace = "legacy";
}
class Legacy_Op<string mnemonic, list<Trait> traits = []> :
    Op<Legacy_Dialect, mnemonic, traits>;

def Results_Dialect : Dialect {
  let name = "results";
  let cppNamespace = "results";
  let useOpFoldResults = 1;
}
class Results_Op<string mnemonic, list<Trait> traits = []> :
    Op<Results_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Dialect without the bit
//===----------------------------------------------------------------------===//

def Legacy_SingleResultOp : Legacy_Op<"single_result"> {
  let results = (outs AnyType:$res);
  let hasFolder = 1;
}

// CHECK-LABEL: legacy::SingleResultOp declarations
// CHECK: ::mlir::OpFoldResult fold(FoldAdaptor adaptor);

def Legacy_TwoResultsOp : Legacy_Op<"two_results"> {
  let results = (outs AnyType:$lhs, AnyType:$rhs);
  let hasFolder = 1;
}

// CHECK-LABEL: legacy::TwoResultsOp declarations
// CHECK: ::llvm::LogicalResult fold(FoldAdaptor adaptor, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);

def Legacy_VariadicResultOp : Legacy_Op<"variadic_result"> {
  let results = (outs Variadic<AnyType>:$res);
  let hasFolder = 1;
}

// CHECK-LABEL: legacy::VariadicResultOp declarations
// CHECK: ::llvm::LogicalResult fold(FoldAdaptor adaptor, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);

//===----------------------------------------------------------------------===//
// Dialect with the bit
//===----------------------------------------------------------------------===//

def Results_MixedResultsOp : Results_Op<"mixed_results"> {
  let results = (outs AnyType:$fixed, Variadic<AnyType>:$variadic);
  let hasFolder = 1;
}

// CHECK-LABEL: results::MixedResultsOp declarations
// CHECK-NOT: SmallVectorImpl<::mlir::OpFoldResult>
// CHECK: ::mlir::OpFoldResults fold(FoldAdaptor adaptor);

def Results_OptionalResultOp : Results_Op<"optional_result"> {
  let results = (outs Optional<AnyType>:$res);
  let hasFolder = 1;
}

// CHECK-LABEL: results::OptionalResultOp declarations
// CHECK-NOT: SmallVectorImpl<::mlir::OpFoldResult>
// CHECK: ::mlir::OpFoldResults fold(FoldAdaptor adaptor);

// The bit does not change the declaration of an op with exactly one fixed
// result.
def Results_SingleResultOp : Results_Op<"single_result"> {
  let results = (outs AnyType:$res);
  let hasFolder = 1;
}

// CHECK-LABEL: results::SingleResultOp declarations
// CHECK: ::mlir::OpFoldResult fold(FoldAdaptor adaptor);

def Results_TwoResultsOp : Results_Op<"two_results"> {
  let results = (outs AnyType:$lhs, AnyType:$rhs);
  let hasFolder = 1;
}

// CHECK-LABEL: results::TwoResultsOp declarations
// CHECK-NOT: SmallVectorImpl<::mlir::OpFoldResult>
// CHECK: ::mlir::OpFoldResults fold(FoldAdaptor adaptor);

def Results_VariadicResultOp : Results_Op<"variadic_result"> {
  let results = (outs Variadic<AnyType>:$res);
  let hasFolder = 1;
}

// CHECK-LABEL: results::VariadicResultOp declarations
// CHECK-NOT: SmallVectorImpl<::mlir::OpFoldResult>
// CHECK: ::mlir::OpFoldResults fold(FoldAdaptor adaptor);

def Results_ZeroResultsOp : Results_Op<"zero_results"> {
  let hasFolder = 1;
}

// CHECK-LABEL: results::ZeroResultsOp declarations
// CHECK-NOT: SmallVectorImpl<::mlir::OpFoldResult>
// CHECK: ::mlir::OpFoldResults fold(FoldAdaptor adaptor);
