//===- Parser.h - MLIR Parser Library Interface -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is contains the interface to the MLIR parser library.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_PARSER_PARSER_H
#define MLIR_PARSER_PARSER_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include <cstddef>

namespace llvm {
class SourceMgr;
class SMDiagnostic;
class StringRef;
} // namespace llvm

namespace mlir {
class AsmParserState;

namespace detail {

/// Given a block containing operations that have just been parsed, if the block
/// contains a single operation of `ContainerOpT` type then remove it from the
/// block and return it. If the block does not contain just that operation,
/// create a new operation instance of `ContainerOpT` and move all of the
/// operations within `parsedBlock` into the first block of the first region.
/// `ContainerOpT` is required to have a single region containing a single
/// block, and must implement the `SingleBlockImplicitTerminator` trait.
template <typename ContainerOpT>
inline OwningOpRef<ContainerOpT> constructContainerOpForParserIfNecessary(
    Block *parsedBlock, MLIRContext *context, Location sourceFileLoc) {
  static_assert(
      ContainerOpT::template hasTrait<OpTrait::OneRegion>() &&
          (ContainerOpT::template hasTrait<OpTrait::NoTerminator>() ||
           OpTrait::template hasSingleBlockImplicitTerminator<
               ContainerOpT>::value),
      "Expected `ContainerOpT` to have a single region with a single "
      "block that has an implicit terminator or does not require one");

  // Check to see if we parsed a single instance of this operation.
  if (llvm::hasSingleElement(*parsedBlock)) {
    if (ContainerOpT op = dyn_cast<ContainerOpT>(parsedBlock->front())) {
      op->remove();
      return op;
    }
  }

  // If not, then build a new one to contain the parsed operations.
  OpBuilder builder(context);
  ContainerOpT op = builder.create<ContainerOpT>(sourceFileLoc);
  OwningOpRef<ContainerOpT> opRef(op);
  assert(op->getNumRegions() == 1 && llvm::hasSingleElement(op->getRegion(0)) &&
         "expected generated operation to have a single region with a single "
         "block");
  Block *opBlock = &op->getRegion(0).front();
  opBlock->getOperations().splice(opBlock->begin(),
                                  parsedBlock->getOperations());

  // After splicing, verify just this operation to ensure it can properly
  // contain the operations inside of it.
  if (failed(op.verifyInvariants()))
    return OwningOpRef<ContainerOpT>();
  return opRef;
}
} // namespace detail

/// This parses the file specified by the indicated SourceMgr and appends parsed
/// operations to the given block. If the block is non-empty, the operations are
/// placed before the current terminator. If parsing is successful, success is
/// returned. Otherwise, an error message is emitted through the error handler
/// registered in the context, and failure is returned. If `sourceFileLoc` is
/// non-null, it is populated with a file location representing the start of the
/// source file that is being parsed. If `asmState` is non-null, it is populated
/// with detailed information about the parsed IR (including exact locations for
/// SSA uses and definitions). `asmState` should only be provided if this
/// detailed information is desired.
LogicalResult parseSourceFile(const llvm::SourceMgr &sourceMgr, Block *block,
                              MLIRContext *context,
                              LocationAttr *sourceFileLoc = nullptr,
                              AsmParserState *asmState = nullptr);

/// This parses the file specified by the indicated filename and appends parsed
/// operations to the given block. If the block is non-empty, the operations are
/// placed before the current terminator. If parsing is successful, success is
/// returned. Otherwise, an error message is emitted through the error handler
/// registered in the context, and failure is returned. If `sourceFileLoc` is
/// non-null, it is populated with a file location representing the start of the
/// source file that is being parsed.
LogicalResult parseSourceFile(llvm::StringRef filename, Block *block,
                              MLIRContext *context,
                              LocationAttr *sourceFileLoc = nullptr);

/// This parses the file specified by the indicated filename using the provided
/// SourceMgr and appends parsed operations to the given block. If the block is
/// non-empty, the operations are placed before the current terminator. If
/// parsing is successful, success is returned. Otherwise, an error message is
/// emitted through the error handler registered in the context, and failure is
/// returned. If `sourceFileLoc` is non-null, it is populated with a file
/// location representing the start of the source file that is being parsed. If
/// `asmState` is non-null, it is populated with detailed information about the
/// parsed IR (including exact locations for SSA uses and definitions).
/// `asmState` should only be provided if this detailed information is desired.
LogicalResult parseSourceFile(llvm::StringRef filename,
                              llvm::SourceMgr &sourceMgr, Block *block,
                              MLIRContext *context,
                              LocationAttr *sourceFileLoc = nullptr,
                              AsmParserState *asmState = nullptr);

/// This parses the IR string and appends parsed operations to the given block.
/// If the block is non-empty, the operations are placed before the current
/// terminator. If parsing is successful, success is returned. Otherwise, an
/// error message is emitted through the error handler registered in the
/// context, and failure is returned. If `sourceFileLoc` is non-null, it is
/// populated with a file location representing the start of the source file
/// that is being parsed.
LogicalResult parseSourceString(llvm::StringRef sourceStr, Block *block,
                                MLIRContext *context,
                                LocationAttr *sourceFileLoc = nullptr);

namespace detail {
/// The internal implementation of the templated `parseSourceFile` methods
/// below, that simply forwards to the non-templated version.
template <typename ContainerOpT, typename... ParserArgs>
inline OwningOpRef<ContainerOpT> parseSourceFile(MLIRContext *ctx,
                                                 ParserArgs &&...args) {
  LocationAttr sourceFileLoc;
  Block block;
  if (failed(parseSourceFile(std::forward<ParserArgs>(args)..., &block, ctx,
                             &sourceFileLoc)))
    return OwningOpRef<ContainerOpT>();
  return detail::constructContainerOpForParserIfNecessary<ContainerOpT>(
      &block, ctx, sourceFileLoc);
}
} // namespace detail

/// This parses the file specified by the indicated SourceMgr. If the source IR
/// contained a single instance of `ContainerOpT`, it is returned. Otherwise, a
/// new instance of `ContainerOpT` is constructed containing all of the parsed
/// operations. If parsing was not successful, null is returned and an error
/// message is emitted through the error handler registered in the context, and
/// failure is returned. `ContainerOpT` is required to have a single region
/// containing a single block, and must implement the
/// `SingleBlockImplicitTerminator` trait.
template <typename ContainerOpT>
inline OwningOpRef<ContainerOpT>
parseSourceFile(const llvm::SourceMgr &sourceMgr, MLIRContext *context) {
  return detail::parseSourceFile<ContainerOpT>(context, sourceMgr);
}

/// This parses the file specified by the indicated filename. If the source IR
/// contained a single instance of `ContainerOpT`, it is returned. Otherwise, a
/// new instance of `ContainerOpT` is constructed containing all of the parsed
/// operations. If parsing was not successful, null is returned and an error
/// message is emitted through the error handler registered in the context, and
/// failure is returned. `ContainerOpT` is required to have a single region
/// containing a single block, and must implement the
/// `SingleBlockImplicitTerminator` trait.
template <typename ContainerOpT>
inline OwningOpRef<ContainerOpT> parseSourceFile(StringRef filename,
                                                 MLIRContext *context) {
  return detail::parseSourceFile<ContainerOpT>(context, filename);
}

/// This parses the file specified by the indicated filename using the provided
/// SourceMgr. If the source IR contained a single instance of `ContainerOpT`,
/// it is returned. Otherwise, a new instance of `ContainerOpT` is constructed
/// containing all of the parsed operations. If parsing was not successful, null
/// is returned and an error message is emitted through the error handler
/// registered in the context, and failure is returned. `ContainerOpT` is
/// required to have a single region containing a single block, and must
/// implement the `SingleBlockImplicitTerminator` trait.
template <typename ContainerOpT>
inline OwningOpRef<ContainerOpT> parseSourceFile(llvm::StringRef filename,
                                                 llvm::SourceMgr &sourceMgr,
                                                 MLIRContext *context) {
  return detail::parseSourceFile<ContainerOpT>(context, filename, sourceMgr);
}

/// This parses the provided string containing MLIR. If the source IR contained
/// a single instance of `ContainerOpT`, it is returned. Otherwise, a new
/// instance of `ContainerOpT` is constructed containing all of the parsed
/// operations. If parsing was not successful, null is returned and an error
/// message is emitted through the error handler registered in the context, and
/// failure is returned. `ContainerOpT` is required to have a single region
/// containing a single block, and must implement the
/// `SingleBlockImplicitTerminator` trait.
template <typename ContainerOpT>
inline OwningOpRef<ContainerOpT> parseSourceString(llvm::StringRef sourceStr,
                                                   MLIRContext *context) {
  LocationAttr sourceFileLoc;
  Block block;
  if (failed(parseSourceString(sourceStr, &block, context, &sourceFileLoc)))
    return OwningOpRef<ContainerOpT>();
  return detail::constructContainerOpForParserIfNecessary<ContainerOpT>(
      &block, context, sourceFileLoc);
}

/// This parses a single MLIR attribute to an MLIR context if it was valid.  If
/// not, an error message is emitted through a new SourceMgrDiagnosticHandler
/// constructed from a new SourceMgr with a single a MemoryBuffer wrapping
/// `attrStr`. If the passed `attrStr` has additional tokens that were not part
/// of the type, an error is emitted.
// TODO: Improve diagnostic reporting.
Attribute parseAttribute(llvm::StringRef attrStr, MLIRContext *context);
Attribute parseAttribute(llvm::StringRef attrStr, Type type);

/// This parses a single MLIR attribute to an MLIR context if it was valid.  If
/// not, an error message is emitted through a new SourceMgrDiagnosticHandler
/// constructed from a new SourceMgr with a single a MemoryBuffer wrapping
/// `attrStr`. The number of characters of `attrStr` parsed in the process is
/// returned in `numRead`.
Attribute parseAttribute(llvm::StringRef attrStr, MLIRContext *context,
                         size_t &numRead);
Attribute parseAttribute(llvm::StringRef attrStr, Type type, size_t &numRead);

/// This parses a single MLIR type to an MLIR context if it was valid.  If not,
/// an error message is emitted through a new SourceMgrDiagnosticHandler
/// constructed from a new SourceMgr with a single a MemoryBuffer wrapping
/// `typeStr`. If the passed `typeStr` has additional tokens that were not part
/// of the type, an error is emitted.
// TODO: Improve diagnostic reporting.
Type parseType(llvm::StringRef typeStr, MLIRContext *context);

/// This parses a single MLIR type to an MLIR context if it was valid.  If not,
/// an error message is emitted through a new SourceMgrDiagnosticHandler
/// constructed from a new SourceMgr with a single a MemoryBuffer wrapping
/// `typeStr`. The number of characters of `typeStr` parsed in the process is
/// returned in `numRead`.
Type parseType(llvm::StringRef typeStr, MLIRContext *context, size_t &numRead);

/// This parses a single IntegerSet to an MLIR context if it was valid. If not,
/// an error message is emitted through a new SourceMgrDiagnosticHandler
/// constructed from a new SourceMgr with a single MemoryBuffer wrapping
/// `str`. If the passed `str` has additional tokens that were not part of the
/// IntegerSet, a failure is returned. Diagnostics are printed on failure if
/// `printDiagnosticInfo` is true.
IntegerSet parseIntegerSet(llvm::StringRef str, MLIRContext *context,
                           bool printDiagnosticInfo = true);

} // namespace mlir

#endif // MLIR_PARSER_PARSER_H
