//===- ComprehensiveBufferize.h - Linalg bufferization pass -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LINALG_TRANSFORMS_COMPREHENSIVE_BUFFERIZE_H
#define MLIR_DIALECT_LINALG_TRANSFORMS_COMPREHENSIVE_BUFFERIZE_H

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/SetOperations.h"

namespace mlir {

class DominanceInfo;
class FuncOp;
class GlobalCreator;

namespace linalg {

/// The BufferizationAliasInfo class maintains a list of buffer aliases and
/// equivalence classes to support bufferization.
/// ExtractSliceOps have special behavior, they act as a level of indirection
/// for bufferization. They don't create reads or writes themselves and analysis
/// needs to look through their uses.
/// ExtractSliceOp + InsertSliceOp have special joint behavior: they may
/// bufferize to the same buffer (i.e. subview), which is what introduces the
/// need for bufferization classes.
/// Some of these functionalities could be refactored in a Bufferizer class that
/// uses BufferizationAliasInfo.
class BufferizationAliasInfo {
public:
  /// Specify fine-grain relationship between buffers to enable more analysis.
  enum class BufferRelation {
    None,
    // TODO: ResultContainsOperand,
    // TODO: OperandContainsResult,
    Equivalent
  };

  explicit BufferizationAliasInfo(Operation *rootOp);

  /// Add a new entry for `v` in the `aliasInfo` and `equivalentInfo`. In the
  /// beginning the alias and equivalence sets only contain `v` itself.
  void createAliasInfoEntry(Value v);

  /// Insert an info entry for `newValue` and merge its alias set with that of
  /// `alias`.
  void insertNewBufferAlias(Value newValue, Value alias);

  /// Insert an info entry for `newValue` and merge its alias set with that of
  /// `alias`. Additionally, merge their equivalence classes.
  void insertNewBufferEquivalence(Value newValue, Value alias);

  /// Return true if the buffer to which `operand` would bufferize aliases a
  /// buffer that is known to not be writable. This implies that the matching
  /// OpResult cannot be bufferized inplace.
  bool aliasesNonWritableBuffer(OpOperand &operand) const;

  /// Return true if the buffer to which `operand` would bufferize is equivalent
  /// to some buffer write.
  bool aliasesInPlaceWrite(Value v) const;

  /// Set the inPlace bufferization spec to true.
  /// Merge result's and operand's aliasing sets and iterate to a fixed point.
  void bufferizeInPlace(OpResult result, OpOperand &operand,
                        BufferRelation bufferRelation = BufferRelation::None);

  /// Set the inPlace bufferization spec to false.
  void bufferizeOutOfPlace(OpResult result);

  /// Return true if it is possible to find an inplace write W among `usesWrite`
  /// and a read R among `usesRead`, such that W and R interfere.
  /// Such a (W, R) pair is an interference to the inplace bufferization of
  /// opResult when:
  ///   1. R is not known properly dominate W (i.e. the effects of the write may
  ///      be visible from R).
  ///   2. one cannot find an intermediate clobbering write `C` to W, such that
  ///      C interleaved between W and R (i.e. W -> C -> R where -> denotes
  ///      dominance).
  bool wouldCreateReadAfterWriteInterference(
      Operation *opToBufferize, DenseSet<OpOperand *> &usesRead,
      DenseSet<OpOperand *> &usesWrite, const DominanceInfo &domInfo) const;

  /// Assume that result bufferizes in-place with one of the operation's
  /// operands. Return true if it is possible to find an inplace write W (resp.
  /// a read R) among the uses of `aliasInfo[result]`, and a read R (resp. an
  /// inplace write W) among the uses of
  /// `aliasInfo[getAliasingOpOperand(result)]`, such that W and R interfere.
  /// Interference detection is needed to determine which cases may bufferize
  /// inplace without interferences. Such cases comprise:
  ///
  /// ```
  ///  %0 = op_to_bufferize(%1)
  ///  read(%1)
  ///
  ///  %0 = op_to_bufferize(%1)
  ///  write(%0)
  ///  read(%1)
  ///
  ///  %0 = op_to_bufferize(%1)
  ///  write(%1)
  ///  read(%0)
  /// ```
  bool
  wouldCreateReadAfterWriteInterference(OpResult result,
                                        const DominanceInfo &domInfo) const;

  /// Return true if `v1` and `v2` bufferize to equivalent buffers.
  bool areEquivalentBufferizedValues(Value v1, Value v2) const {
    return equivalentInfo.getLeaderValue(v1) ==
           equivalentInfo.getLeaderValue(v2);
  }

  /// Return true if the source of an `insertSliceOp` bufferizes to an
  /// equivalent ExtractSliceOp.
  bool isSourceEquivalentToAMatchingInplaceExtractSliceOp(
      tensor::InsertSliceOp insertSliceOp) const;

  /// Apply `fun` to all the members of the equivalence class of `v`.
  void applyOnEquivalenceClass(Value v, function_ref<void(Value)> fun) const;

  /// Return true if the value is known to bufferize to writable memory.
  bool bufferizesToWritableMemory(Value v) const;

  /// Specify that the value is known to bufferize to writable memory.
  void setBufferizesToWritableMemory(Value v);

  /// Print to `os`.
  void printAliases(raw_ostream &os) const;
  void printEquivalences(raw_ostream &os) const;

  /// Print to `errs()`.
  void dumpAliases() const;
  void dumpEquivalences() const;

private:
  /// llvm::EquivalenceClasses wants comparable elements because it uses
  /// std::set as the underlying impl.
  /// ValueWrapper wraps Value and uses pointer comparison on the defining op.
  /// This is a poor man's comparison but it's not like UnionFind needs ordering
  /// anyway ..
  struct ValueWrapper {
    ValueWrapper(Value val) : v(val) {}
    operator Value() const { return v; }
    bool operator<(const ValueWrapper &wrap) const {
      return v.getImpl() < wrap.v.getImpl();
    }
    bool operator==(const ValueWrapper &wrap) const { return v == wrap.v; }
    Value v;
  };

  using EquivalenceClassRangeType = llvm::iterator_range<
      llvm::EquivalenceClasses<ValueWrapper>::member_iterator>;
  /// Check that aliasInfo for `v` exists and return a reference to it.
  EquivalenceClassRangeType getAliases(Value v) const;

  /// Return true if the (ExtractSliceOp, InsertSliceOp) pair match (i.e.
  /// equivalent operand / result and same offset/sizes/strides specification).
  ///
  /// This is one particular type of relationship between ops on tensors that
  /// reduce to an equivalence on buffers. This should be generalized and
  /// exposed as interfaces on the proper types.
  bool areEquivalentExtractSliceOps(tensor::ExtractSliceOp st,
                                    tensor::InsertSliceOp sti) const;

  /// Return true if there is a `candidateOp` that would write to memory after
  /// bufferization and such that:
  ///   1. The written buffer is equivalent to either `aliasingRead` or
  ///      `aliasingWrite` under the inPlace bufferization decisions taken
  ///      so far.
  ///   2. `aliasingWrite` properly dominates `candidateOp`.
  ///   3. `candidateOp` properly dominates `aliasingReadOp`.
  // TODO: richer clobbering analysis with container-containee relationship
  // instead of equivalence.
  bool existsInterleavedValueClobber(OpOperand &aliasingRead,
                                     OpOperand &aliasingWrite,
                                     const DominanceInfo &domInfo) const;

  /// Return true if there is a write that:
  ///   1. Properly dominates aliasingReadOp.
  ///   2. Is properly dominated by aliasingWriteOp.
  ///   3. Clobbers the write that would be interfering with the read.
  ///
  /// Case discussion:
  /// ================
  /// Case 1: opOperand is produced by opToBufferize,
  /// Case 2: opResult is produced by opToBufferize,
  /// Common case:
  ///   - aliasingReadOp is a read to an alias of opOperand.
  ///   - aliasingWriteOp is an inplace write to an alias of opResult.
  ///   - aliasingWriteOp dominates aliasingReadOp.
  ///
  /// ```
  ///    // Either case 1:
  ///    %opOperand = opToBufferize(%opResult)
  ///    aliasingWriteOp(%aliasingWrite = alias(%opResult)) // inplace
  ///     aliasingReadOp( %aliasingRead = alias(%opOperand))
  /// ```
  ///
  /// ```
  ///    // Or case 2:
  ///    %opResult = opToBufferize(%opOperand)
  ///    aliasingWriteOp(%aliasingWrite = alias(%opResult)) // inplace
  ///     aliasingReadOp( %aliasingRead = alias(%opOperand))
  /// ```
  ///
  /// Capture possible cases where `aliasingWriteOp(alias(%opResult))` has no
  /// visible effect on `aliasingReadOp(alias(%opOperand))`.
  bool isClobberedWriteBeforeRead(Operation *opToBufferize,
                                  OpOperand &aliasingRead,
                                  OpOperand &aliasingWrite,
                                  const DominanceInfo &domInfo) const;

  /// Set of tensors that are known to bufferize to writable memory.
  llvm::DenseSet<Value> bufferizeToWritableMemory;

  /// Auxiliary structure to store all the values a given value aliases with.
  /// These are the conservative cases that can further decompose into
  /// "equivalent" buffer relationships.
  llvm::EquivalenceClasses<ValueWrapper> aliasInfo;

  /// Auxiliary structure to store all the equivalent buffer classes.
  llvm::EquivalenceClasses<ValueWrapper> equivalentInfo;
};

/// Analyze the `ops` to determine which OpResults are inplaceable:
LogicalResult inPlaceAnalysis(SmallVector<Operation *> &ops,
                              BufferizationAliasInfo &aliasInfo,
                              const DominanceInfo &domInfo);

/// Bufferize one particular op.
/// `bufferizedFunctionTypes` (resp. `globalCreator`) are expected to be
/// non-null if `op` is a CallOpInterface (resp. GlobalCreator).
LogicalResult
bufferizeOp(Operation *op, BlockAndValueMapping &bvm,
            BufferizationAliasInfo &aliasInfo,
            DenseMap<FuncOp, FunctionType> *bufferizedFunctionTypes = nullptr,
            GlobalCreator *globalCreator = nullptr);

} // namespace linalg
} // namespace mlir

#endif // define MLIR_DIALECT_LINALG_TRANSFORMS_COMPREHENSIVE_BUFFERIZE_H
