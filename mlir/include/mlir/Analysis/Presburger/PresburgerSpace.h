//===- PresburgerSpace.h - MLIR PresburgerSpace Class -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Classes representing space information like number of variables and kind of
// variables.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_PRESBURGER_PRESBURGERSPACE_H
#define MLIR_ANALYSIS_PRESBURGER_PRESBURGERSPACE_H

#include "mlir/Support/TypeID.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/PointerLikeTypeTraits.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace presburger {

/// Kind of variable. Implementation wise SetDims are treated as Range
/// vars, and spaces with no distinction between dimension vars are treated
/// as relations with zero domain vars.
enum class VarKind { Symbol, Local, Domain, Range, SetDim = Range };

/// PresburgerSpace is the space of all possible values of a tuple of integer
/// valued variables/variables. Each variable has one of the three types:
///
/// Dimension: Ordinary variables over which the space is represented.
///
/// Symbol: Symbol variables correspond to fixed but unknown values.
/// Mathematically, a space with symbolic variables is like a
/// family of spaces indexed by the symbolic variables.
///
/// Local: Local variables correspond to existentially quantified variables.
/// For example, consider the space: `(x, exists q)` where x is a dimension
/// variable and q is a local variable. Let us put the constraints:
///       `1 <= x <= 7, x = 2q`
/// on this space to get the set:
///       `(x) : (exists q : q <= x <= 7, x = 2q)`.
/// An assignment to symbolic and dimension variables is valid if there
/// exists some assignment to the local variable `q` satisfying these
/// constraints. For this example, the set is equivalent to {2, 4, 6}.
/// Mathematically, existential quantification can be thought of as the result
/// of projection. In this example, `q` is existentially quantified. This can be
/// thought of as the result of projecting out `q` from the previous example,
/// i.e. we obtained {2, 4, 6} by projecting out the second dimension from
/// {(2, 1), (4, 2), (6, 2)}.
///
/// Dimension variables are further divided into Domain and Range variables
/// to support building relations.
///
/// Variables are stored in the following order:
///       [Domain, Range, Symbols, Locals]
///
/// A space with no distinction between types of dimension variables can
/// be implemented as a space with zero domain. VarKind::SetDim should be used
/// to refer to dimensions in such spaces.
///
/// Compatibility of two spaces implies that number of variables of each kind
/// other than Locals are equal. Equality of two spaces implies that number of
/// variables of each kind are equal.
///
/// PresburgerSpace optionally also supports attaching attachments to each
/// variable in space. `resetAttachments<AttachmentType>` enables attaching
/// attachments to space. All attachments must be of the same type,
/// `AttachmentType`. `AttachmentType` must have a
/// `llvm::PointerLikeTypeTraits` specialization available and should be
/// supported via mlir::TypeID.
///
/// These attachments can be used to check if two variables in two different
/// spaces correspond to the same variable.
class PresburgerSpace {
public:
  static PresburgerSpace getRelationSpace(unsigned numDomain = 0,
                                          unsigned numRange = 0,
                                          unsigned numSymbols = 0,
                                          unsigned numLocals = 0) {
    return PresburgerSpace(numDomain, numRange, numSymbols, numLocals);
  }

  static PresburgerSpace getSetSpace(unsigned numDims = 0,
                                     unsigned numSymbols = 0,
                                     unsigned numLocals = 0) {
    return PresburgerSpace(/*numDomain=*/0, /*numRange=*/numDims, numSymbols,
                           numLocals);
  }

  unsigned getNumDomainVars() const { return numDomain; }
  unsigned getNumRangeVars() const { return numRange; }
  unsigned getNumSetDimVars() const { return numRange; }
  unsigned getNumSymbolVars() const { return numSymbols; }
  unsigned getNumLocalVars() const { return numLocals; }

  unsigned getNumDimVars() const { return numDomain + numRange; }
  unsigned getNumDimAndSymbolVars() const {
    return numDomain + numRange + numSymbols;
  }
  unsigned getNumVars() const {
    return numDomain + numRange + numSymbols + numLocals;
  }

  /// Get the number of vars of the specified kind.
  unsigned getNumVarKind(VarKind kind) const;

  /// Return the index at which the specified kind of var starts.
  unsigned getVarKindOffset(VarKind kind) const;

  /// Return the index at Which the specified kind of var ends.
  unsigned getVarKindEnd(VarKind kind) const;

  /// Get the number of elements of the specified kind in the range
  /// [varStart, varLimit).
  unsigned getVarKindOverlap(VarKind kind, unsigned varStart,
                             unsigned varLimit) const;

  /// Return the VarKind of the var at the specified position.
  VarKind getVarKindAt(unsigned pos) const;

  /// Insert `num` variables of the specified kind at position `pos`.
  /// Positions are relative to the kind of variable. Return the absolute
  /// column position (i.e., not relative to the kind of variable) of the
  /// first added variable.
  ///
  /// If attachments are being used, the newly added variables have no
  /// attachments.
  unsigned insertVar(VarKind kind, unsigned pos, unsigned num = 1);

  /// Removes variables of the specified kind in the column range [varStart,
  /// varLimit). The range is relative to the kind of variable.
  void removeVarRange(VarKind kind, unsigned varStart, unsigned varLimit);

  /// Swaps the posA^th variable of kindA and posB^th variable of kindB.
  void swapVar(VarKind kindA, VarKind kindB, unsigned posA, unsigned posB);

  /// Returns true if both the spaces are compatible i.e. if both spaces have
  /// the same number of variables of each kind (excluding locals).
  bool isCompatible(const PresburgerSpace &other) const;

  /// Returns true if both the spaces are equal including local variables i.e.
  /// if both spaces have the same number of variables of each kind (including
  /// locals).
  bool isEqual(const PresburgerSpace &other) const;

  /// Changes the partition between dimensions and symbols. Depending on the new
  /// symbol count, either a chunk of dimensional variables immediately before
  /// the split become symbols, or some of the symbols immediately after the
  /// split become dimensions.
  void setVarSymbolSeperation(unsigned newSymbolCount);

  void print(llvm::raw_ostream &os) const;
  void dump() const;

  //===--------------------------------------------------------------------===//
  //     Attachment Interactions
  //===--------------------------------------------------------------------===//

  /// Set the attachment for `i^th` variable to `attachment`. `T` here should
  /// match the type used to enable attachments.
  template <typename T>
  void setAttachment(VarKind kind, unsigned i, T attachment) {
#ifdef LLVM_ENABLE_ABI_BREAKING_CHECKS
    assert(TypeID::get<T>() == attachmentType && "Type mismatch");
#endif
    atAttachment(kind, i) =
        llvm::PointerLikeTypeTraits<T>::getAsVoidPointer(attachment);
  }

  /// Get the attachment for `i^th` variable casted to type `T`. `T` here
  /// should match the type used to enable attachments.
  template <typename T>
  T getAttachment(VarKind kind, unsigned i) const {
#ifdef LLVM_ENABLE_ABI_BREAKING_CHECKS
    assert(TypeID::get<T>() == attachmentType && "Type mismatch");
#endif
    return llvm::PointerLikeTypeTraits<T>::getFromVoidPointer(
        atAttachment(kind, i));
  }

  /// Check if the i^th variable of the specified kind has a non-null
  /// attachment.
  bool hasAttachment(VarKind kind, unsigned i) const {
    return atAttachment(kind, i) != nullptr;
  }

  /// Check if the spaces are compatible, as well as have the same attachments
  /// for each variable.
  bool isAligned(const PresburgerSpace &other) const;
  /// Check if the number of variables of the specified kind match, and have
  /// same attachments with the other space.
  bool isAligned(const PresburgerSpace &other, VarKind kind) const;

  /// Find the variable of the specified kind with attachment `val`.
  /// PresburgerSpace::kIdNotFound if attachment is not found.
  template <typename T>
  unsigned findId(VarKind kind, T val) const {
    unsigned i = 0;
    for (unsigned e = getNumVarKind(kind); i < e; ++i)
      if (hasAttachment(kind, i) && getAttachment<T>(kind, i) == val)
        return i;
    return kIdNotFound;
  }
  static const unsigned kIdNotFound = UINT_MAX;

  /// Returns if attachments are being used.
  bool isUsingAttachments() const { return usingAttachments; }

  /// Reset the stored attachments in the space. Enables `usingAttachments` if
  /// it was `false` before.
  template <typename T>
  void resetAttachments() {
    attachments.clear();
    attachments.resize(getNumDimAndSymbolVars());
#ifdef LLVM_ENABLE_ABI_BREAKING_CHECKS
    attachmentType = TypeID::get<T>();
#endif

    usingAttachments = true;
  }

  /// Disable attachments being stored in space.
  void disableAttachments() {
    attachments.clear();
    usingAttachments = false;
  }

protected:
  PresburgerSpace(unsigned numDomain = 0, unsigned numRange = 0,
                  unsigned numSymbols = 0, unsigned numLocals = 0)
      : numDomain(numDomain), numRange(numRange), numSymbols(numSymbols),
        numLocals(numLocals) {}

  void *&atAttachment(VarKind kind, unsigned i) {
    assert(usingAttachments &&
           "Cannot access attachments when `usingAttachments` is false.");
    assert(kind != VarKind::Local &&
           "Local variables cannot have attachments.");
    return attachments[getVarKindOffset(kind) + i];
  }

  void *atAttachment(VarKind kind, unsigned i) const {
    assert(usingAttachments &&
           "Cannot access attachments when `usingAttachments` is false.");
    assert(kind != VarKind::Local &&
           "Local variables cannot have attachments.");
    return attachments[getVarKindOffset(kind) + i];
  }

private:
  // Number of variables corresponding to domain variables.
  unsigned numDomain;

  // Number of variables corresponding to range variables.
  unsigned numRange;

  /// Number of variables corresponding to symbols (unknown but constant for
  /// analysis).
  unsigned numSymbols;

  /// Number of variables corresponding to locals (variables corresponding
  /// to existentially quantified variables).
  unsigned numLocals;

  /// Stores whether or not attachments are being used in this space.
  bool usingAttachments = false;

#ifdef LLVM_ENABLE_ABI_BREAKING_CHECKS
  /// TypeID of the attachments in space. This should be used in asserts only.
  TypeID attachmentType;
#endif

  /// Stores a attachment for each non-local variable as a `void` pointer.
  SmallVector<void *, 0> attachments;
};

} // namespace presburger
} // namespace mlir

#endif // MLIR_ANALYSIS_PRESBURGER_PRESBURGERSPACE_H
