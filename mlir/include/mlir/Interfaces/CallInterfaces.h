//===- CallInterfaces.h - Call Interfaces for MLIR --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the definitions of the call interfaces defined in
// `CallInterfaces.td`.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INTERFACES_CALLINTERFACES_H
#define MLIR_INTERFACES_CALLINTERFACES_H

#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/PointerUnion.h"

namespace mlir {
/// A callable is either a symbol, or an SSA value, that is referenced by a
/// call-like operation. This represents the destination of the call.
struct CallInterfaceCallable : public PointerUnion<SymbolRefAttr, Value> {
  using PointerUnion<SymbolRefAttr, Value>::PointerUnion;
};

class CallOpInterface;

namespace call_interface_impl {

/// Resolve the callable operation for given callee to a CallableOpInterface, or
/// nullptr if a valid callable was not resolved.  `symbolTable` is an optional
/// parameter that will allow for using a cached symbol table for symbol lookups
/// instead of performing an O(N) scan.
Operation *resolveCallable(CallOpInterface call,
                           SymbolTableCollection *symbolTable = nullptr);

/// Parse a function or call result list.
///
///   function-result-list ::= function-result-list-parens
///                          | non-function-type
///   function-result-list-parens ::= `(` `)`
///                                 | `(` function-result-list-no-parens `)`
///   function-result-list-no-parens ::= function-result (`,` function-result)*
///   function-result ::= type attribute-dict?
///
ParseResult
parseFunctionResultList(OpAsmParser &parser, SmallVectorImpl<Type> &resultTypes,
                        SmallVectorImpl<DictionaryAttr> &resultAttrs);

/// Parses a function signature using `parser`. This does not deal with function
/// signatures containing SSA region arguments (to parse these signatures, use
/// function_interface_impl::parseFunctionSignature). When
/// `mustParseEmptyResult`, `-> ()` is expected when there is no result type.
///
///   no-ssa-function-signature ::= `(` no-ssa-function-arg-list `)`
///                               -> function-result-list
///   no-ssa-function-arg-list  ::= no-ssa-function-arg
///                               (`,` no-ssa-function-arg)*
///   no-ssa-function-arg       ::= type attribute-dict?
ParseResult parseFunctionSignature(OpAsmParser &parser,
                                   SmallVectorImpl<Type> &argTypes,
                                   SmallVectorImpl<DictionaryAttr> &argAttrs,
                                   SmallVectorImpl<Type> &resultTypes,
                                   SmallVectorImpl<DictionaryAttr> &resultAttrs,
                                   bool mustParseEmptyResult = true);

/// Print a function signature for a call or callable operation. If a body
/// region is provided, the SSA arguments are printed in the signature. When
/// `printEmptyResult` is false, `-> function-result-list` is omitted when
/// `resultTypes` is empty.
///
///   function-signature     ::= ssa-function-signature
///                            | no-ssa-function-signature
///   ssa-function-signature ::= `(` ssa-function-arg-list `)`
///                            -> function-result-list
///   ssa-function-arg-list  ::= ssa-function-arg (`,` ssa-function-arg)*
///   ssa-function-arg       ::= `%`name `:` type attribute-dict?
void printFunctionSignature(OpAsmPrinter &p, TypeRange argTypes,
                            ArrayAttr argAttrs, bool isVariadic,
                            TypeRange resultTypes, ArrayAttr resultAttrs,
                            Region *body = nullptr,
                            bool printEmptyResult = true);

/// Adds argument and result attributes, provided as `argAttrs` and
/// `resultAttrs` arguments, to the list of operation attributes in `result`.
/// Internally, argument and result attributes are stored as dict attributes
/// with special names given by getResultAttrName, getArgumentAttrName.
void addArgAndResultAttrs(Builder &builder, OperationState &result,
                          ArrayRef<DictionaryAttr> argAttrs,
                          ArrayRef<DictionaryAttr> resultAttrs,
                          StringAttr argAttrsName, StringAttr resAttrsName);
void addArgAndResultAttrs(Builder &builder, OperationState &result,
                          ArrayRef<OpAsmParser::Argument> args,
                          ArrayRef<DictionaryAttr> resultAttrs,
                          StringAttr argAttrsName, StringAttr resAttrsName);

} // namespace call_interface_impl

} // namespace mlir

namespace llvm {

// Allow llvm::cast style functions.
template <typename To>
struct CastInfo<To, mlir::CallInterfaceCallable>
    : public CastInfo<To, mlir::CallInterfaceCallable::PointerUnion> {};

template <typename To>
struct CastInfo<To, const mlir::CallInterfaceCallable>
    : public CastInfo<To, const mlir::CallInterfaceCallable::PointerUnion> {};

} // namespace llvm

/// Include the generated interface declarations.
#include "mlir/Interfaces/CallInterfaces.h.inc"

#endif // MLIR_INTERFACES_CALLINTERFACES_H
