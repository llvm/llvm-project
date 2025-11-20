//===--------------- Ripple.h - Expand RIpple intrinsics ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass expands RIpple intrinsics.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_VECTORIZE_RIPPLE_H
#define LLVM_TRANSFORMS_VECTORIZE_RIPPLE_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/DomTreeUpdater.h"
#include "llvm/Analysis/MemorySSA.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/SimplifyQuery.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Use.h"
#include "llvm/IR/User.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/Support/Error.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <numeric>
#include <utility>

namespace llvm {

class Argument;
class Constant;
class DataLayout;
class IntegerType;
class IntrinsicInst;
class MDNode;
class MemoryLocation;
class TargetMachine;
class Value;
class raw_ostream;

/// @brief Tensor shape (shape w/ fixed number of dimensions)
/// @tparam SizeTy The type used to store dimension sizes
///
/// The dimensions are ordered from innermost (dimension 0) to outermost.
template <typename SizeTy> class TensorShapeAny {
private:
  /// @brief Base size of vector structures of this object
  static constexpr unsigned BaseTensorSize = 4;

public:
  using DimSize = SizeTy;
  using Shape = SmallVector<DimSize, BaseTensorSize>;
  using const_iterator = typename Shape::const_iterator;
  using const_reverse_iterator = typename Shape::const_reverse_iterator;

  /// @brief A scalar!
  TensorShapeAny() = default;

  /// @brief Constructor moving a shape
  TensorShapeAny(Shape &&s) : shape(std::move(s)) {}

  TensorShapeAny(unsigned Rank, DimSize Val = DimSize(1)) : shape(Rank, Val) {}

  /// @brief Constructor from a container (with begin and end methods)
  /// @param shape the shape
  TensorShapeAny(ArrayRef<DimSize> &s) : shape(s) {}

  /// @brief Constructor from iterators
  /// @tparam It iterator type
  /// @param sBegin the beginning of the shape values
  /// @param sEnd the ending of the shape values
  template <typename It, typename It2,
            typename = EnableIfConvertibleToInputIterator<It>>
  TensorShapeAny(It sBegin, It sEnd) : shape(sBegin, sEnd) {}

  /// @brief Constructor with a set rank and unique dimension set to size
  /// @param rank the rank of the tensor
  /// @param dim the dimension to set the size
  /// @param size the size
  TensorShapeAny(unsigned rank, unsigned dim, DimSize size)
      : shape(rank, DimSize(1)) {
    shape[dim] = size;
  }

  /// @brief Getter for the vector shape of this value
  /// @return the vector shape
  const Shape &getShape() const { return shape; }

  /// @brief Accessor for the shape at a given index
  /// @param index the index
  /// @return the shape value at the specified index
  /// N.B. If the index is greater than the rank, a size of 1 is returned
  inline DimSize operator[](size_t index) const {
    return index < shape.size() ? shape[index] : DimSize(1);
  }

  /// @brief Check if the value is scalar
  /// @return True if this shape is scalar, false otherwise
  bool isScalar() const {
    return !std::any_of(shape.begin(), shape.end(),
                        [](auto &val) { return val > DimSize(1); });
  }

  /// @brief Check if the value is a vector
  /// @return True if this shape is a vector, false otherwise
  bool isVector() const { return !isScalar(); }

  /// @brief Returns the number of dimensions of this shape (rank)
  /// @return the rank of this shape
  unsigned rank() const { return shape.size(); }

  /// @brief Returns a flat (1D) shape of this value
  /// @return the flat shape
  DimSize flatShape() const {
    return std::accumulate(begin(), end(), DimSize(1),
                           std::multiplies<DimSize>());
  };

  /// @brief An iterator starting w/ the innermost dimension shape
  const_iterator begin() const { return shape.begin(); }
  const_iterator end() const { return shape.end(); }

  /// @brief A reverse iterator starting w/ the outermost dimension shape
  const_reverse_iterator rbegin() const { return shape.rbegin(); }
  const_reverse_iterator rend() const { return shape.rend(); }

  /// @brief Prints this shape
  /// @param O the output stream
  void print(raw_ostream &O) const;

  /// @brief Equality operator
  /// @return true if the shapes match, false otherwise
  /// N.B.: The shapes do not need to be of the same rank, only non-empty
  /// dimensions are checked for equality.
  bool operator==(const TensorShapeAny<DimSize> &other) const;

  /// @brief Inequality operator
  /// @see operator==()
  inline bool operator!=(const TensorShapeAny<DimSize> &other) const {
    return !this->operator==(other);
  }

  /// @brief Lexicographical ordering of shapes, from higher to lower dimensions
  ///
  /// For example TensorShape[32][2] > TensorShape[32][1] >
  /// TensorShape[31][4000]
  bool operator<(const TensorShapeAny<DimSize> &other) const;
  bool operator>(const TensorShapeAny<DimSize> &other) const;
  bool operator<=(const TensorShapeAny<DimSize> &other) const;
  bool operator>=(const TensorShapeAny<DimSize> &other) const;

  /// @brief Combines the shape of other into *this* to construct a broadcast
  /// shape.
  /// @param other the other shape.
  /// @return An error string or success.
  Error combineShapeBcast(const TensorShapeAny<DimSize> &other);

  /// @brief Checks that this shape can be combined with other.
  /// @param other The shape to be broadcasted to.
  /// @return A string error message or success if it can be broadcasted.
  Error canCombineWith(const TensorShapeAny<DimSize> &other) const;

  /// @brief Checks that this shape can be broadcasted to other.
  /// @param other The shape to be broadcasted to.
  /// @return A string error message or success if it can be broadcasted.
  Error isBroadcastError(const TensorShapeAny<DimSize> &other) const;

  /// @brief Reduces *this* for the indices set in the BitVector
  /// @param bv the dimensions to be reduced
  /// @returns true if any dimension was reduced, false otherwise
  bool reduceDimensions(const BitVector &bv);

  /// @brief Keeps only the dimensions set in the BitVector
  /// @param bv the dimensions to keep
  /// @returns true if any dimension was removed, false otherwise
  bool keepDimensions(const BitVector &bv);

  /// @brief Returns true if the shape is scalar after being reduced along the
  /// dimensions specified by reduction.
  /// @param reduction The set of dimensions to be reduced
  /// @return True if scalar after reduction, false otherwise
  bool reducedToScalarBy(const BitVector &reduction) const;

  /// @brief Returns an offset, in number of elements from the start of the
  /// Tensor, for the given multi-dimensional access vector.
  ///
  /// We allow indices to be out of bound for dimensions that are empty (size
  /// 1). This is useful to compute mappings from an index from a shape that is
  /// a broadcast of self and the index coming from the shape itself to compute
  /// a shuffle mask.
  ///
  /// @param coordinates Index for which the offset is computed
  /// @return the offset
  size_t getOffsetAt(ArrayRef<size_t> coordinate) const;

  /// @brief Calls the function *f* for each valid coordinate of *this* shape
  void foreachIndex(std::function<void(ArrayRef<size_t>)> f) const;

  /// @brief bit vector representing the dimensions
  ///        whose size is greater than one.
  BitVector nonEmptyDims() const;

  /// @brief applies a test to all dimensions of this tensor shape and \p other,
  ///        and reports the result as a BitVector.
  /// @param test the test to apply to this tensor's dimensions and
  ///             \p other 's dimensions (one by one)
  BitVector
  testBothDims(const TensorShapeAny<DimSize> &other,
               const std::function<bool(DimSize, DimSize)> &test) const;

  /// @brief Returns the bitset of the dimensions that are simulatenously non-1
  /// in both this shape as well as in \p other.
  /// @param other another shape
  /// @return the dimensions that are non empty in both this and other.
  BitVector bothNonEmptyDims(const TensorShapeAny<DimSize> &other) const;

  /// @brief Returns the bitset of dimensions of this that needs to be reduced
  /// before a broadcast to other is feasible.
  /// @param other another shape
  /// @return the dimensions that needs to be reduced to enable broadcasting
  /// this to other
  BitVector reductionDimensionsBeforeBroadcast(
      const TensorShapeAny<DimSize> &other) const;

  /// @brief dimensions of this tensor shape that would need a splat/broadcast
  ///        to match up those of \p other
  BitVector requiredSplat(const TensorShapeAny<SizeTy> &other) const;

  /// Create a metadata node to store this tensor shape
  MDNode *toConstMetadata(IntegerType *Ty) const;

  /// Retrieve a tensor shape if it's rank is lower or equal to Rank, nullptr
  /// otherwise or if the metadata is not present
  static std::unique_ptr<TensorShapeAny<SizeTy>>
  fromConstMetadata(unsigned Rank, const MDNode *Node);

  /// @brief Returns the result of broadcasting all the references to
  /// TensorShapes in the provided iterator. Returns an Error if the shapes are
  /// incompatible
  /// @param AllToBcast an iterator over references to TensorShapeAny
  template <typename IteratorT>
  static Expected<TensorShapeAny<DimSize>>
  broadcastShapeFromAll(llvm::iterator_range<IteratorT> AllToBcast);
  static Expected<TensorShapeAny<DimSize>>
  broadcastShapeFromAll(ArrayRef<const TensorShapeAny<DimSize> *> AllToBcast);

private:
  /// @brief Tensor shape; the innermost dimension starts at index 0
  Shape shape;
  Error checkDims(const TensorShapeAny<DimSize> &other,
                  std::function<Error(unsigned Idx, DimSize, DimSize)>) const;
};

using TensorShape = TensorShapeAny<uint64_t>;

/// @brief Convenience arrow operator to print TensorShapeAny
inline raw_ostream &operator<<(raw_ostream &OS, const TensorShape &tshape) {
  tshape.print(OS);
  return OS;
}

class Ripple {
  TargetMachine *TM;

public:
  using PEIdentifier = unsigned;
  using PEIndex = unsigned;
  using TensorIndex = unsigned;
  using DimSize = TensorShape::DimSize;
  using RippleIntrinsicIndex = std::pair<PEIdentifier, PEIndex>;
  static constexpr unsigned RippleIntrinsicsMaxDims = 10u;
  enum DimType {
    VectorDimension,
    ThreadDimension,
    CoreDimension,
    DeviceDimension,
    UnknownDimType,
  };
  enum ProcessingStatus {
    ShapePropagationFailure,
    WaitingForSpecialization,
    SemanticsCheckFailure,
    Success,
  };

  /// @brief Sole constructor
  /// @param F a function to process
  /// @param AM the analysis manager to get the relevant analysis from
  Ripple(TargetMachine *TM, Function &F, FunctionAnalysisManager &AM,
         ArrayRef<std::pair<PEIdentifier, DimType>> dimensionTypes,
         ProcessingStatus &PS,
         DenseSet<AssertingVH<Function>> &SpecializationsPending,
         DenseSet<AssertingVH<Function>> &SpecializationsAvailable)
      : TM(TM), F(F), DL(F.getParent()->getDataLayout()),
        targetLibraryInfo(AM.getResult<TargetLibraryAnalysis>(F)),
        targetTransformInfo(AM.getResult<TargetIRAnalysis>(F)),
        domTree(AM.getResult<DominatorTreeAnalysis>(F)),
        postdomTree(AM.getResult<PostDominatorTreeAnalysis>(F)),
        MemSSA(AM.getResult<MemorySSAAnalysis>(F).getMSSA()),
        MemSSAWalker(*MemSSA.getWalker()), AA(MemSSA.getAA()),
        DTU(domTree, postdomTree, DomTreeUpdater::UpdateStrategy::Lazy),
        irBuilder(F.getContext()), PERanks(gatherRippleFunctionUsage(F)),
        TensorDimIDMap(buildPEIdMap()),
        AssumptionCache(AM.getResult<AssumptionAnalysis>(F)),
        SQ(DL, &targetLibraryInfo, &domTree, &AssumptionCache),
        ScalarShape(TensorShape(tensorRank())), PS(PS),
        SpecializationsPending(SpecializationsPending),
        SpecializationsAvailable(SpecializationsAvailable) {
    // Set the types
    for (const auto &pair : dimensionTypes) {
      idTypes.insert(pair);
    }
  }

  ~Ripple() { delete FuncRPOT; }

  /// @brief Ripple cannot be copied!
  Ripple(const Ripple &) = delete;

  /// @brief Builds the vector of PE identifier for each tensor dimension
  SmallVector<PEIdentifier, 8> buildPEIdMap();

  /// @brief Initializes \ref Ripple::FuncRPOT
  void initFuncRPOT();

  /// @brief Propagate shapes throughout the function
  Error propagateShapes(bool &WaitingForSpecialization);

  /// @brief Returns the rank of the tensors used by Ripple.
  /// @return the rank of Ripple tensors
  inline size_t tensorRank() const { return TensorDimIDMap.size(); };

  /// @brief Prints scalar instruction's that were promoted to Tensors with
  /// their shape and types
  /// @param os the output stream
  void printTensorInstructions(raw_ostream &os) const;

  /// @brief Convenience function to query the shape of Instruction, Constants
  /// or Arguments.
  /// @param ShapePropagation When true, during shape propagation, returns
  /// ShapeIgnoredByRipple for Constants and Arguments having vector types
  const TensorShape &getRippleShape(const Value *V,
                                    bool ShapePropagation = false) const;

  /// @brief Sets a tensor shape to a value
  /// @return true if the shape of v was modified
  bool setRippleShape(const Value *V, const TensorShape &Shape);

  /// @brief Returns a copy of \ref Ripple::InstructionRippleShapes.
  /// \note The caller is expected to call \ref Ripple::propagateShapes before
  /// calling this method.
  std::map<AssertingVH<const Instruction>, TensorShape>
  getInstructionToRippleShape() const;

private:
  /// @brief The function we are working on
  Function &F;

  const DataLayout &DL;

  /// @brief Get the target vector capabilities info
  TargetLibraryAnalysis::Result &targetLibraryInfo;

  /// @brief Get the target transform info
  TargetTransformInfo &targetTransformInfo;

  /// @brief Dominator tree
  DominatorTreeAnalysis::Result &domTree;

  /// @brief Post-dominator tree
  PostDominatorTreeAnalysis::Result &postdomTree;

  /// @brief MemorySSA & AliasAnalysis for Alloca
  MemorySSA &MemSSA;
  MemorySSAWalker &MemSSAWalker;
  AliasAnalysis &AA;

  /// @brief Helper to update domTree & postDomTree
  DomTreeUpdater DTU;

  /// @brief The reverse post-order of BBs in F
  ReversePostOrderTraversal<Function *> *FuncRPOT = nullptr;

  /// @brief An IR builder!
  IRBuilder<> irBuilder;

  /// @brief A map between a PE identifier and its type
  DenseMap<PEIdentifier, DimType> idTypes;

  /// @brief A mapping between a PE identifier and the number of
  /// dimensions that are accessed in the function
  /// @see gatherRippleFunctionUsage
  const DenseMap<PEIdentifier, DimSize> PERanks;

  /// @brief A map between a tensor dimension and it's ripple id
  const SmallVector<PEIdentifier, 8> TensorDimIDMap;

  /// @brief A mapping between IR Instructions to a ripple shape
  std::map<AssertingVH<const Instruction>, TensorShape> InstructionRippleShapes;

  /// @brief A cache used by SimplifyQuery
  AssumptionAnalysis::Result &AssumptionCache;

  /// @brief Used to simplify expressions generated by LinearSeries w/ PHINodes
  SimplifyQuery SQ;

  /// @brief A convenience scalar shape
  const TensorShape ScalarShape;

  /// @brief A shape for values that should't be modified by Ripple, also has
  /// the nice property to be the neutral shape for broadcasting!
  const TensorShape &ShapeIgnoredByRipple = ScalarShape;

  DenseSet<AssertingVH<AllocaInst>> PromotableAlloca;
  DenseSet<AssertingVH<AllocaInst>> NonPromotableAlloca;

  /// @brief A way to return the processing status of this function pass to the
  /// Ripple module pass
  ProcessingStatus &PS;

  /// @brief The pending and available specialization sets
  DenseSet<AssertingVH<Function>> &SpecializationsPending,
      &SpecializationsAvailable;

  /// @brief Initializes *idRanks* with data extracted from ripple intrinsics in
  /// the function
  static DenseMap<PEIdentifier, DimSize>
  gatherRippleFunctionUsage(const Function &F);

  /// @brief CallInsts that are inside a vector conditional region and requires
  /// masking
  DenseSet<AssertingVH<const CallInst>> MaskedCalls;

  /// @brief Look for the block shape corresponding to a struct
  /// "ripple_block_shape*".
  /// The template parameter is used to specialize the function for checking
  /// ripple semantics (true) and fast access (false)
  template <bool = false>
  IntrinsicInst *getBlockShapeIntrinsic(const Use &RippleBlockShapePtr);

  /// @brief ensor index from a ripple index
  /// @param idAndIdx A PE id and index
  /// @return the tensor dimension index
  /// @see dimIdAndIndex
  TensorIndex rippleToTensor(const RippleIntrinsicIndex &idAndIdx) const;

  /// @brief Returns the ripple PE id and index from a tensor index
  /// @param tensorDimIdx the tensor dimension index
  /// @return a pair<Ripple ID, dimension index>
  RippleIntrinsicIndex tensorToRipple(TensorIndex tensorIdx) const;

  /// @brief checks if the given identifier belongs to a vector dimension
  /// @return true if it is a vector dimension, false otherwise
  inline bool isVectorId(PEIdentifier id) const {
    return idType(id) == VectorDimension;
  }

  /// @brief Returns the type of a ripple dimension ID
  /// @param id the ripple dimension id
  /// @return the dimension type
  inline DimType idType(PEIdentifier id) const {
    auto iter = idTypes.find(id);
    if (iter != idTypes.end())
      return iter->second;
    else
      return UnknownDimType;
  }

  /// @brief Returns the type of a tensor dimension
  /// @param tensorDimIdx the t
  DimType tensorDimType(TensorIndex tensorIdx) const {
    return idType(tensorToRipple(tensorIdx).first);
  }

  /// @brief Returns the rank of a PE
  /// @param id the identifier
  /// @return its rank (number of dimensions)
  DimSize PERank(PEIdentifier PEId) const;

  /// @brief Sets a tensor shape for instruction I
  /// @return true if the shape of v was modified
  bool setRippleShape(const Instruction *I, const TensorShape &Shape);

  /// @brief Returns the tensor shape of I.
  /// The shape of an instruction is the result of propagating tensor shapes
  /// throughout the function. Hence an instruction with scalar type may have a
  /// tensor shape.
  const TensorShape &getRippleShape(const Instruction *I) const;

  /// @brief Only scalar constant can be queried for shape (Scalar)
  /// @param ShapePropagation When true, during shape propagation, returns
  /// ShapeIgnoredByRipple for Constants having vector types
  const TensorShape &getRippleShape(const Constant *C,
                                    bool ShapePropagation = false) const;

  /// @brief Only scalar arguments can be queried for shape (Scalar)
  /// This can be improved in the future when we implement the vector function
  /// interface.
  /// @param ShapePropagation When true, during shape propagation, returns
  /// ShapeIgnoredByRipple for Arguments having vector types
  const TensorShape &getRippleShape(const Argument *C,
                                    bool ShapePropagation = false) const;

  /// @brief Returns an IntrinsicInst pointer if I is a ripple.dim,
  /// ripple.dim.size or ripple.dim.setsize intrinsics
  /// @param I an instruction
  /// @return a pointer if I is ripple.dim or ripple.dim.size, nullptr otherwise
  static IntrinsicInst *rippleBlockIntrinsics(Instruction *I);
  static const IntrinsicInst *rippleBlockIntrinsics(const Instruction *I) {
    return rippleBlockIntrinsics(const_cast<Instruction *>(I));
  }

  /// @brief Returns all intrinsics that are Ripple reductions
  /// @param I an instruction
  /// @return a pointer if I is a Ripple reduction, nullptr otherwise
  static IntrinsicInst *rippleReduceIntrinsics(Instruction *I);
  static const IntrinsicInst *rippleReduceIntrinsics(const Instruction *I) {
    return rippleReduceIntrinsics(const_cast<Instruction *>(I));
  }

  /// @brief Returns all intrinsics that are Ripple shuffles
  /// @param I an instruction
  /// @return a pointer if I is a Ripple reduction, nullptr otherwise
  static IntrinsicInst *rippleShuffleIntrinsics(Instruction *I);
  static const IntrinsicInst *rippleShuffleIntrinsics(const Instruction *I) {
    return rippleShuffleIntrinsics(const_cast<Instruction *>(I));
  }

  /// @brief Returns all intrinsics that are Ripple rotate to lower
  /// instructions.
  /// @param I an instruction
  /// @return a pointer if I is a Ripple reduction, nullptr otherwise
  static IntrinsicInst *rippleRotToLowerIntrinsics(Instruction *I);
  static const IntrinsicInst *rippleRotToLowerIntrinsics(const Instruction *I) {
    return rippleRotToLowerIntrinsics(const_cast<Instruction *>(I));
  }

  /// @brief Returns a ripple broadcast intrinsic, or nullptr if I is not
  static IntrinsicInst *rippleBroadcastIntrinsic(Instruction *I);
  static const IntrinsicInst *rippleBroadcastIntrinsic(const Instruction *I) {
    return rippleBroadcastIntrinsic(const_cast<Instruction *>(I));
  }

  /// @brief Returns a ripple slice intrinsic, or nullptr if I is not
  static IntrinsicInst *rippleSliceIntrinsic(Instruction *I);
  static const IntrinsicInst *rippleSliceIntrinsic(const Instruction *I) {
    return rippleSliceIntrinsic(const_cast<Instruction *>(I));
  }

  /// @brief Returns the intrinsics that take the block shape as operands
  static IntrinsicInst *rippleIntrinsicsWithBlockShapeOperand(Instruction *I);
  static const IntrinsicInst *
  rippleIntrinsicsWithBlockShapeOperand(const Instruction *I) {
    return rippleIntrinsicsWithBlockShapeOperand(const_cast<Instruction *>(I));
  }

  /// @brief Returns true if I is any of the Ripple intrinsic call
  static bool isRippleIntrinsics(const Instruction *I);

  /// @brief Strips inlining debug location from ripple intrinsics call,
  /// otherwise return the instruction's debug location untouched
  static DebugLoc sanitizeRippleLocation(const Instruction *I);

  /// @brief Returns the dimensions that are reduced by Ripple reductions
  /// intrinsics. *this method only works after shape propagation*.
  /// @param I any instruction
  /// @return the bitset of ripple dimensions index to reduce
  BitVector reductionTensorDimensions(const IntrinsicInst *I) const;

  /// @brief last tensor dimension of @p IShape that corresponds to a
  /// non-trivial vector
  ///        (e.g. SIMD) processing element block dimension.
  ///        Creates a diagnostic message if none is found.
  /// @param I the intrinsic associated with block shape @p IShape
  /// @param SpecialArgIdx an argument index for @p I that we want to show
  ///                      in error messages
  ///                      if we don't find an applicable dimension
  /// @param OpKind a string that represents the kind of intrinsic we are
  ///               inspecting - for error messaging as well.
  unsigned lastVectorIdx(const IntrinsicInst *I, const TensorShape &IShape,
                         const unsigned SpecialArgIdx,
                         const char *OpKind) const;

  /// @brief Computes the shape of the ReductionI instruction given the shape of
  /// it's input tensor.
  ///
  /// This is only useful for shape inference, use getRippleShape(ReductionI)
  /// instead or reductionTensorDimensions to get the bit set of reduced
  /// dimensions.
  ///
  /// @param ReductionI the reduction instruction
  /// @param InputShape the shape of the input tensor
  /// @return

  /// @brief Returns the output shape of Ripple intrinsics w/ bitsets (reduce
  /// and broadcast) from the shape of the input tensor and bitset.
  Expected<TensorShape>
  computeRippleShapeForBitsetIntrinsic(const IntrinsicInst *I,
                                       const TensorShape &IShape);

  /// @brief Tests if all the reachable instruction of the function have an
  /// associated Ripple shape
  /// @return true if all instructions have a ripple shape, false otherwise
  bool allInstructionsHaveRippleShapes() const;

  /// @brief Returns the set of basic blocks ]from, to] by doing a depth-first
  /// search starting at @p from and stopping when encountering @p to. @p to
  /// must post-dominate @p from.
  DenseSet<BasicBlock *> allBasicBlocksFromTo(BasicBlock *from,
                                              BasicBlock *to) const;

  /// @brief Returns the range of operands that can be vectorized
  ///
  /// For call instructions, only the call arguments are visited
  /// For branch/switch only the condition is visited
  /// For every other Instruction kinds, all operands are visited
  static iterator_range<User::const_op_iterator>
  vectorizableOperands(const Instruction *I);

  /// @see vectorizableOperands(const Instruction *I)
  static iterator_range<User::op_iterator> vectorizableOperands(Instruction *I);

  /// @brief computes the expected shape of an instruction from its operands
  /// When AllowPartialPhi is true, return a partial shape for PHI nodes.
  Expected<TensorShape>
  inferShapeFromOperands(const Instruction *I, bool AllowPartialPhi,
                         bool &RequiresWaitingForSpecialization);

  /// @brief Returns true if there are no vector dimensions, false otherwise
  bool hasNoVectorDimension() const;

  /// @brief Returns the broadcast of @p ShapeToBeBroadcasted and @p OtherShape
  /// or warns the user that the broadcast could not happen and returns an Error
  Expected<TensorShape> combineShapeBcastWithErrorReporting(
      const TensorShape &ShapeToBeBroadcasted, const TensorShape &OtherShape,
      StringRef ShapeToBeBcastedMsg, DebugLoc ShapeToBeBroadcastedLocation,
      StringRef OtherShapeMsg = StringRef(),
      DebugLoc SecondLocation = DebugLoc());

  /// @brief Visits all the MemoryUse that use the memory of @p Def through any
  /// possible program path
  /// @param Apply The function being applied to apply to MemoryUse's
  /// instructions being affected by @p Def. The function must return true to
  /// continue to the next Use or false to stop early.
  void
  visitAllInstructionsBeingClobberedBy(const MemoryDef *Def,
                                       std::function<bool(Instruction *)> Apply,
                                       bool VisitUnreachableFromEntry = false);

  /// @brief Visits all the MemoryDef that we are sure (in must Aliasing)
  /// clobber this @p Use
  /// @param Apply The function to call to the MemoryDef's instruction that
  /// clobbers @p Use. The function must return true to continue to the next Use
  /// or false to stop early.
  void visitAllClobberingInstructions(MemoryUse *Use,
                                      std::function<bool(Instruction *)> Apply,
                                      bool VisitUnreachableFromEntry = false);

  /// @brief Checks if the @p Loc MemoryLocation aliases with any of the
  /// alloca in @p AllocaSet
  std::pair<AliasResult, AllocaInst *>
  aliasesWithAlloca(const MemoryLocation &Loc,
                    const DenseSet<AssertingVH<AllocaInst>> &AllocaSet,
                    AliasResult::Kind AliasKind) const;

  /// @brief Checks if the @p Loc MemoryLocation aliases with alloca promotable
  /// by Ripple
  AllocaInst *aliasesWithPromotableAlloca(const MemoryLocation &Loc) const {
    return aliasesWithAlloca(Loc, PromotableAlloca, AliasResult::MustAlias)
        .second;
  }

  /// @brief Checks if the @p Loc MemoryLocation aliases with alloca that are
  /// not promotable by Ripple
  AllocaInst *
  aliasesWith_Non_PromotableAlloca(const MemoryLocation &Loc) const {
    // Return if any alias kind exists
    return aliasesWithAlloca(Loc, NonPromotableAlloca, AliasResult::NoAlias)
        .second;
  }

  /// @brief The reverse post-order of BBs in F. Note that if \ref FuncRPOT is
  /// invalid this method performs the reverse post order traversal analysis.
  /// This is analysis is expensive, hence invocation of this method comes with
  /// a performance advisory.
  const ReversePostOrderTraversal<Function *> &getFuncRPOT() {
    if (FuncRPOT == nullptr)
      initFuncRPOT();
    return *FuncRPOT;
  }

  /// @brief Invalidates FuncRPOT member variable, accesses to this->FuncRPOT
  /// will lead to a runtime error. Caller must invoke \ref initFuncRPOT
  /// manually to repopulate FuncRPOT with valid state.
  void invalidateFuncRPOT() {
    delete FuncRPOT;
    FuncRPOT = nullptr;
  }

  /// @brief Returns the tensor shape of RippleSetShape
  TensorShape setShapeToTensorShape(const IntrinsicInst *RippleSetShape) const;

  /// @brief Checks that all the ripple instrinsics in this function that rely
  /// on a block size can find them and returns an error otherwise.
  Error checkBlockShapeUsage(const Function &F);

  /// @brief Ripple vectorizes calls when they have at least one tensor argument
  /// with vector shapes and no argument with regular (LLVM) vector type
  /// arguments.
  bool rippleVectorizeCall(const CallInst &CI) const;
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_VECTORIZE_RIPPLE_H
