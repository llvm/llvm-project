//===- IR/OpenMPIRBuilder.h - OpenMP encoding builder for LLVM IR - C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the OpenMPIRBuilder class and helpers used as a convenient
// way to create LLVM instructions for OpenMP directives.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OPENMP_IR_IRBUILDER_H
#define LLVM_OPENMP_IR_IRBUILDER_H

#include "llvm/IR/DebugLoc.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Frontend/OpenMP/OMPConstants.h"

namespace llvm {

/// An interface to create LLVM-IR for OpenMP directives.
///
/// Each OpenMP directive has a corresponding public generator method.
class OpenMPIRBuilder {
public:
  /// Create a new OpenMPIRBuilder operating on the given module \p M. This will
  /// not have an effect on \p M (see initialize).
  OpenMPIRBuilder(Module &M) : M(M), Builder(M.getContext()) {}

  /// Initialize the internal state, this will put structures types and
  /// potentially other helpers into the underlying module. Must be called
  /// before any other method and only once!
  void initialize();

  /// Add attributes known for \p FnID to \p Fn.
  void addAttributes(omp::RuntimeFunction FnID, Function &Fn);

  /// Set the cancellation block to \p CBB.
  void setCancellationBlock(BasicBlock *CBB) { CancellationBlock = CBB; }

  /// Type used throughout for insertion points.
  using InsertPointTy = IRBuilder<>::InsertPoint;

  /// Description of a LLVM-IR insertion point (IP) and a debug/source location
  /// (filename, line, column, ...).
  struct LocationDescription {
    template <typename T, typename U>
    LocationDescription(const IRBuilder<T, U> &IRB)
        : IP(IRB.saveIP()), DL(IRB.getCurrentDebugLocation()) {}
    LocationDescription(const InsertPointTy &IP) : IP(IP) {}
    LocationDescription(const InsertPointTy &IP, const DebugLoc &DL)
        : IP(IP), DL(DL) {}
    InsertPointTy IP;
    DebugLoc DL;
  };

  /// Emitter methods for OpenMP directives.
  ///
  ///{

  /// Generator for '#omp barrier'
  ///
  /// \param Loc The location where the barrier directive was encountered.
  /// \param DK The kind of directive that caused the barrier.
  /// \param ForceSimpleCall Flag to force a simple (=non-cancellation) barrier.
  /// \param CheckCancelFlag Flag to indicate a cancel barrier return value
  ///                        should be checked and acted upon.
  ///
  /// \returns The insertion point after the barrier.
  InsertPointTy CreateBarrier(const LocationDescription &Loc, omp::Directive DK,
                              bool ForceSimpleCall = false,
                              bool CheckCancelFlag = true);

  ///}

private:
  /// Update the internal location to \p Loc.
  bool updateToLocation(const LocationDescription &Loc) {
    Builder.restoreIP(Loc.IP);
    Builder.SetCurrentDebugLocation(Loc.DL);
    return Loc.IP.getBlock() != nullptr;
  }

  /// Return the function declaration for the runtime function with \p FnID.
  Function *getOrCreateRuntimeFunction(omp::RuntimeFunction FnID);

  /// Return the (LLVM-IR) string describing the source location \p LocStr.
  Constant *getOrCreateSrcLocStr(StringRef LocStr);

  /// Return the (LLVM-IR) string describing the default source location.
  Constant *getOrCreateDefaultSrcLocStr();

  /// Return the (LLVM-IR) string describing the source location \p Loc.
  Constant *getOrCreateSrcLocStr(const LocationDescription &Loc);

  /// Return an ident_t* encoding the source location \p SrcLocStr and \p Flags.
  Value *getOrCreateIdent(Constant *SrcLocStr,
                          omp::IdentFlag Flags = omp::IdentFlag(0));

  /// Generate a barrier runtime call.
  ///
  /// \param Loc The location at which the request originated and is fulfilled.
  /// \param DK The directive which caused the barrier
  /// \param ForceSimpleCall Flag to force a simple (=non-cancellation) barrier.
  /// \param CheckCancelFlag Flag to indicate a cancel barrier return value
  ///                        should be checked and acted upon.
  ///
  /// \returns The insertion point after the barrier.
  InsertPointTy emitBarrierImpl(const LocationDescription &Loc,
                                omp::Directive DK, bool ForceSimpleCall,
                                bool CheckCancelFlag);

  /// Return the current thread ID.
  ///
  /// \param Ident The ident (ident_t*) describing the query origin.
  Value *getOrCreateThreadID(Value *Ident);

  /// The underlying LLVM-IR module
  Module &M;

  /// The LLVM-IR Builder used to create IR.
  IRBuilder<> Builder;

  /// TODO: Stub for a cancellation block stack.
  BasicBlock *CancellationBlock = nullptr;

  /// Map to remember source location strings
  StringMap<Constant *> SrcLocStrMap;

  /// Map to remember existing ident_t*.
  DenseMap<std::pair<Constant *, uint64_t>, GlobalVariable *> IdentMap;
};

} // end namespace llvm

#endif // LLVM_IR_IRBUILDER_H
