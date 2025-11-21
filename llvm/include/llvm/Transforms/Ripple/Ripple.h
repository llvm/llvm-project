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
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/iterator.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/DomTreeUpdater.h"
#include "llvm/Analysis/MemorySSA.h"
#include "llvm/Analysis/MemorySSAUpdater.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/SimplifyQuery.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Use.h"
#include "llvm/IR/User.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/TypeSize.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace llvm {

class Argument;
class DataLayout;
class IntrinsicInst;
class MDNode;
class MemoryLocation;
class TargetMachine;
class DbgRecord;
class DIBuilder;
class DILocalVariable;

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

// _____________________________________________________________________________

/// @brief In one dimension, an affine series f is defined as f(x) = ax + b,
///  where
///  - b, an integer Value, is called the _base_;
///  - a, an integer value as well, is called the _slope_;
///  - x is a vector of size s representing the sequence [0, 1, ..., s - 1].
///
/// For instance, {base = 42, slope = 1, shape = [8]} represents
//// the vector [42, 43, 44, 45, 46, 47, 48, 49] of size 8 .
///
/// Affine series here are extended to a multi-dimensional version, in which
///  - the base b can be a rank-r tensor (of integer Values)
///  - the slope a can be a r-dimensional vector (of integer Values);
///  - x is a set of r [0, 1, ... , s_{r-1}] sequences.
///
/// The weighted sum of all r sequences in x with a as weights
/// forms a r-dimensional tensor L defined by:
/// L(I) = sum_{k=1 to r}(a_k.x_k(I)),
/// for any r-dimensional vector I in [0, s_0)x[0, s1)x...x[0, s_{r-1}).
///
/// And the affine series T is then defined by T(I) = L(I) + b(I).
///
/// We need to keep L pure, because we use it to determine
/// memory access strides (in particular element-size strides)
/// to reveal coalesced loads and stores.
///
/// For that reason, we enforce an invariant that prevents all the nice and
/// regular information in L from intermixing too much with b.
/// We want any tensor dimension in T to be contributed
/// exclusively by either L or b.
/// Let's call trivial dimensions of a tensor the dimensions along which
/// the shape size is 1,
/// and non-trivial dimensions the ones where it's greater than 1.
/// The invariant ("orthogonality") we enforce is that
/// the set of non-trivial dimension of the base b
/// and the set of non-trivial dimensions of L don't intersect.
///
/// A linear series forms a tensor T of rank r is then defined as
/// T(I) = L(I_L) + b(I_b)
/// Where I_L and I_b correspond to the non-trivial indices for (resp.) L and b
///
/// Here are a few more examples
///
/// 1- {base = [x, y], L = {slope = [1, 0], shape = [2, 1]}}
/// can be instantiated as the following matrix:
/// [x    , y]
/// [x + 1, y + 1]
/// [x + 2, y + 2]
/// This could represent (for example) the addresses of
/// the 3 first elements of columns x and y in a matrix.
/// Note that base is a 1x2 tensor here.
///
/// 2- Let A be an address.
/// {base = A, L = {slope = {1, 1024}, shape = [1024, 32]}}.
/// represents the series of contiguous addresses from A to A + 32767.
/// Feed these addresses to a load and you get a large contiguous/coalesced
/// load.
///
/// We use the shapes associated with the slope and the base to determine:
/// - which dimensions of the whole Affine Series are contributed
///   by the slope or the base
/// - the number of elements in the slope (which is a bounded linear series).
///
/// This class is implemented using a TensorShape for the base and the
/// slope, which (as we know) are orthogonal to each other.
/// Hence, we store a scalar
/// slope for each dimension of the slope's shape that is not empty (size > 1)
/// and zero for the others.
// TODO: rename AffineSeries
class LinearSeries : public RefCountedBase<LinearSeries> {
public:
  /// @brief LinearSeries constructor
  /// @param v The base value
  /// @param vShape the base shape
  /// @param s the slope (one for each dimension of the slope shape)
  /// @param sShape the slope shape
  template <typename It>
  LinearSeries(Value *v, const TensorShape &bShape, It s,
               const TensorShape &sShape)
      : Base(v), baseShape(bShape), SlopeValues(s.begin(), s.end()),
        slopeShape(sShape) {
    assert(SlopeValues.size() == slopeShape.rank());
    computeLSShape();
  }
  /// @brief Linear series constructor w/ TensorShape move semantics
  template <typename It>
  LinearSeries(Value *v, TensorShape &&vShape, It s, TensorShape &&sShape)
      : Base(v), baseShape(std::move(vShape)), SlopeValues(s.begin(), s.end()),
        slopeShape(std::move(sShape)) {
    assert(SlopeValues.size() == slopeShape.rank());
    computeLSShape();
  }

  /// @brief Prints this linear series to the given stream
  void print(raw_ostream &O) const;

  /// @brief Returns the shape of the base
  const TensorShape &getBaseShape() const { return baseShape; }

  /// @brief Returns the shape of the slope
  const TensorShape &getSlopeShape() const { return slopeShape; }

  /// @brief Returns the shape of the linear series
  const TensorShape &getShape() const { return LSShape; }

  /// @brief shape along dimension i
  TensorShape::DimSize getShape(unsigned i) const {
    assert(i < rank() && "Index overflow in shape()");
    auto b = baseShape[i];
    auto s = slopeShape[i];
    assert(b > 0 && s > 0);
    assert((b == 1 || s == 1) && "Incompatible base and slope shapes");
    return b > s ? b : s;
  }

  /// @brief Returns the base value of this LinearSeries
  Value *getBase() const { return Base; }

  /// @brief Returns the i-th slope of this LinearSeries
  /// @param DimIndex the dimension index
  Value *getSlope(size_t DimIndex) const { return SlopeValues[DimIndex]; };

  /// @brief Returns a range of modifiable slopes of this LinearSeries
  auto slopes() { return make_range(SlopeValues.begin(), SlopeValues.end()); }

  /// @brief Returns a range of all the slopes of this LinearSeries
  auto slopes() const {
    return make_range(SlopeValues.begin(), SlopeValues.end());
  }

  /// @brief the rank of the tensor underlying this linear series
  unsigned rank() const { return getSlopeShape().rank(); }

  /// @brief Dimensions along which slopes effectively represent a broadcast.
  /// Basically dimensions for which slopes have a non-1 shape and are 0.
  BitVector getSplatDims() const {
    int r = slopeShape.rank();
    BitVector splats(r);
    for (int i = 0; i < r; ++i) {
      if (Constant *constSlope = dyn_cast<Constant>(getSlope(i))) {
        if (slopeShape[i] > 1 && constSlope->isZeroValue()) {
          splats.set(i);
        }
      }
    }
    return splats;
  }

  /// @brief Checks if the linear series is equal to its base
  bool hasSlope() const { return !slopeShape.isScalar(); }

  /// @brief Whether this affine series is a scalar (base and slope are scalars)
  bool isScalar() const {
    return slopeShape.isScalar() && baseShape.isScalar();
  }

  /// @brief Returns true if all the slopes are zero or the slope shape is
  /// scalar
  bool hasZeroSlopes() const;

  /// @brief Whether this affine series is a scalar or a splat of a scalar
  bool isScalarOrSplat() const;

  static IntegerType *getSlopeTypeFor(const DataLayout &DL, Type *BaseType);

  /// @brief Sets the i-th slope of this linear series to V
  void setSlope(size_t i, Value *V) { SlopeValues[i] = V; };

  /// @brief Set the base to V
  void setBase(Value *V) { Base = V; };

  /// @brief returns a new linear series from this one,
  /// where the slope shape is collapsed (to 1) along dimensions specified by
  /// @p trivialDims. The other attributes are shared with this one.
  LinearSeries removeSlopes(BitVector &trivialDims) const {
    TensorShape NewSlopeShape(getSlopeShape());
    NewSlopeShape.reduceDimensions(trivialDims);
    return LinearSeries(Base, baseShape, SlopeValues, NewSlopeShape);
  }

  /// @brief Constructs a constant linear series of slope 1 and starting value
  /// zero: [0, 1, ..., size - 1]
  ///
  /// @param intTy the type of the linear series's scalar element
  /// @param size the size of the linear series
  static Constant *constructLinearSeriesVector(IntegerType *intTy,
                                               uint64_t size);

private:
  /// @brief The base value
  TrackingVH<Value> Base;
  /// @brief The shape of the base
  TensorShape baseShape;
  /// @brief Slopes (always at least 1 element to store the type)
  SmallVector<TrackingVH<Value>, 3> SlopeValues;
  /// @brief The shape that is contributed by this linear series
  TensorShape slopeShape;
  /// @brief the LS shape
  TensorShape LSShape;

  void computeLSShape() {
    LSShape = baseShape;
    // This should't happen by construction
    Error e = LSShape.combineShapeBcast(slopeShape);
    if (e) {
      llvm_unreachable("Base and slope shapes are incompatible");
    }
  }
};

// _____________________________________________________________________________

/// @brief A data class for n-dimensional load / store attributes
struct NDLoadStoreAttr {
public:
  /// @brief whether each of the memory accesses constituting
  ///        this n-d memory access are aligned.
  bool Aligned;

  /// @brief whether the load or store is masked.
  ///        The mask is always the last n-d load/store argument
  bool Masked;

  /// @brief  dimensions along which a broadcast / splat is performed along
  ///         with the load / store.
  BitVector SplatDims;

  /// @brief dimensions associated with strides.
  ///        A stride paarameter corresponds to each of them,
  ///        in increasing dimensions.
  ///        Note: here we use "stride" as opposed to "slope",
  ///        because it represents a memory stride (in bytes).
  BitVector StrideDims;

  /// @brief when the base addresses of a n-d load or store are represented in
  ///        a tensor, baseDims represents the dimensions of that base
  ///        tensor that are non-1.
  BitVector BaseDims;

  NDLoadStoreAttr(LinearSeries &AddressSeries, BitVector &StrideDims)
      : SplatDims(AddressSeries.getSplatDims()), StrideDims(StrideDims),
        BaseDims(AddressSeries.getBaseShape().nonEmptyDims()) {}

  NDLoadStoreAttr() = default;

  /// @brief Prints the attribute
  void print(raw_ostream &O) const;

  /// @brief Converts a string to an attribute set
  static NDLoadStoreAttr fromString(StringRef AttrName);

  /// @brief True if no attributes are set, false otherwise
  bool empty() const {
    return !Aligned && !Masked && SplatDims.empty() && StrideDims.empty() &&
           BaseDims.empty();
  }
};

inline raw_ostream &operator<<(raw_ostream &OS, const NDLoadStoreAttr &LSAttr) {
  LSAttr.print(OS);
  return OS;
}

// Fwd decl
class Ripple;

/// @brief A factory for calls to the
///        Ripple multi-dimensional load and store API.
/// In the tensor expansion phase of its algorithm, Ripple turns scalar loads
/// and stores into a tensorial (n-dimensional) version.
/// These go through the whole algorithm, and their equivalent in terms of
/// traditional (vector) loads and stores is rendered at the end.
/// This allows us to work on a more homogeneous representation of operations
/// throughout the algorithm.
// The naming convention is:
// nd_load.1d.attr for a 1-d load with attributes
// nd_load.2d for a 2-d load w/o attributes
// nd_store.3d.attr for a 3-d store w/ attributes, etc.
// The syntax of the 'attr' part of the name is defined in NDLoadStoreAttr
class NDLoadStoreFactory {
public:
  NDLoadStoreFactory(IRBuilder<> &IrBuilder, Module &Mod, Ripple &MyRipple)
      : IrBuilder(IrBuilder), Mod(Mod), MyRipple(MyRipple) {};

  /// @brief Generates a n-d load of a data set described by AddressSeries
  /// @param AddressSeries represents the sequence of addresses to be loaded
  ///        into a tensor value
  /// @param ToShape the expected shape of that tensor value
  Value *genLoad(LoadInst *Load, LinearSeries &AddressSeries,
                 Value *DefaultAddress, const TensorShape &ToShape);

  /// @brief Generates a n-d store into a data set described by AddressSeries
  /// @param AddressSeries represents the sequence of addresses to be stored to
  /// @param Val the tensor value to store
  /// @param ToShape the expected shape of that tensor
  Value *genStore(StoreInst *Store, Value *Val, LinearSeries &AddressSeries,
                  Value *DefaultAddress, const TensorShape &ToShape);

  /// @brief Generates a generic, structure-less vector scatter
  ///        of @p Val to @p Address.
  /// @param Store the original store instruction to be replaced by the scatter
  /// @param ToShape the expected shape of the input tensor type.
  /// TODO: Make this private and only leave genStore as public interface
  Value *genUnstructuredStore(StoreInst *Load, Value *Val, Value *Address,
                              const TensorShape &ToShape);

private:
  /// @brief Generates a generic, structure-less vector gather from @p Address.
  /// @param Load the original load instruction to be replaced by the gather
  /// @param ToShape the expected shape of the output tensor type.
  Value *genUnstructuredLoad(LoadInst *Load, Value *Address,
                             const TensorShape &ToShape);

  /// @brief Applies mask @p mask to @p NdLoadStore, a n-d load or store.
  /// If \pNdLoadStore is already masked, use the and-combination of its
  /// current mask with @p NdLoadStore 's mask.
  Value *applyMask(Value *NdLoadStore, Value *mask);

  /// Returns a fixed true mask of @p NElements elements.
  Value *getTrueMask(int NElements) {
    ElementCount FixedNElements = ElementCount::getFixed(NElements);
    return ConstantVector::getSplat(FixedNElements,
                                    ConstantInt::getTrue(Mod.getContext()));
  }

  /// Asserts that @p Val is a fixed vector of @p NElements elements.
  void assertNumElements(Value *Val, int NElements) {
    ElementCount FixedNElements = ElementCount::getFixed(NElements);
    Type *ValType = Val->getType();
    assert(isa<VectorType>(ValType) && "Expecting a vector\n");
    VectorType *ValVectorType = cast<VectorType>(ValType);
    if (ValVectorType->getElementCount() != FixedNElements) {
      errs() << "Expected a vector of " << NElements << " but has value "
             << *Val << " of type " << *ValType << "\n";
      llvm_unreachable("Vector type with wrong element count");
    }
  }

  /// @brief Underlying IR Builder used to generate n-d loads and stores
  IRBuilder<> &IrBuilder;

  /// @brief Underlying module
  Module &Mod;

  /// @brief Ripple instance associated with us
  Ripple &MyRipple;

  // Checks if a series of addresses corresponds to a contiguous memory zone.
  // @param type of the elements of the tensor being addressed
  // @param AddressSeries a linear series representing the sequence of addresses
  //        being accessed
  // @param StrideDims (output) will indicate which
  //        dimensions have regular non-unit strides, according to AddressSeries
  bool analyzeCoalescing(Type *ElementTy, LinearSeries &AddressSeries,
                         BitVector &StrideDims);

  /// @brief Generates a n-d load of a data set described by AddressSeries,
  ///        With the assumption that AddressSeries doesn't contain a splat /
  ///        broadcast.
  /// @param AddressSeries represents the sequence of addresses to be loaded
  ///        into a tensor value
  /// @param ToShape the expected shape of that tensor value
  Value *genLoadNoSplat(LoadInst *Load, LinearSeries &AddressSeries,
                        Value *DefaultAddress, const TensorShape &ToShape);

  /// Naming convention for the n-d load-store API
  static std::string ndFunctionName(StringRef LoadOrStore, int nDims,
                                    NDLoadStoreAttr &Attr);
};

// ---------------------------------------------------------------------------//
//                            Helper Functions                                //
// ---------------------------------------------------------------------------//

/**
 * @brief Updates phi instructions in \p BBToProcess such that all incoming
 * edges from \p RefToRemove are removed.
 *
 * @param RefToRemove The predecessor to remove.
 * @param BBToProcess
 *
 * \note \p RefToRemove might still have \p BBToProcess as the successor. It is
 * the caller's responsibility to appropriately update the successors of \p
 * RefToRemove.
 */
void removeIncomingBlockFromPhis(const BasicBlock *RefToRemove,
                                 BasicBlock *BBToProcess);

// _____________________________________________________________________________

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
        ScalarShape(TensorShape(tensorRank())),
        NdLoadStoreFac(irBuilder, *F.getParent(), *this), PS(PS),
        SpecializationsPending(SpecializationsPending),
        SpecializationsAvailable(SpecializationsAvailable) {
    // Set the types
    for (const auto &pair : dimensionTypes) {
      idTypes.insert(pair);
    }
  }

  ~Ripple() {
    clearLinearSeriesCache();
    delete FuncRPOT;
  }

  /// @brief Validate that Ripple intrinsics throughout the function are
  /// consistent with the specification.
  /// @return Error::success if the specification is followed, an error
  /// otherwise
  Error validate() const;

  /// @brief Ripple cannot be copied!
  Ripple(const Ripple &) = delete;

  /// @brief Builds the vector of PE identifier for each tensor dimension
  SmallVector<PEIdentifier, 8> buildPEIdMap();

  /// @brief Initializes \ref Ripple::FuncRPOT
  void initFuncRPOT();

  /// @brief Replace ripple.get.size by their values
  /// Checks ripple.get.block.size semantics, returning an error when detecting
  /// one
  Error replaceRippleGetSize();

  /// @brief Propagate shapes throughout the function
  Error propagateShapes(bool &WaitingForSpecialization);

  /// @brief Returns the rank of the tensors used by Ripple.
  /// @return the rank of Ripple tensors
  inline size_t tensorRank() const { return TensorDimIDMap.size(); };

  /// @brief Prints scalar instruction's that were promoted to Tensors with
  /// their shape and types
  /// @param os the output stream
  void printTensorInstructions(raw_ostream &os) const;

  /// @brief Prints scalar instruction's that were promoted to Linear Series
  /// @param os the output stream
  void printValidSeries(raw_ostream &oss) const;

  /// @brief Multi-dimensional tensor broadcast of value *V* of shape
  /// *fromShape* to *toShape*.
  Expected<Value *> tensorBcast(Value *V, const TensorShape &FromShape,
                                const TensorShape &ToShape);

  /// @brief Creates a tensorized name for a variable
  ///
  /// An example output is "name.t32x12" for a TensorShape[32][12]
  /// or "Name.ripple.tensor.8" for a TensorShape[8] or "Name.ripple" if
  /// Shape.rank() == 0.
  static std::string tensorizedName(StringRef Name, const TensorShape &Shape);

  /// @brief Non-cached instantiation of a linear series
  Value *instantiateLinearSeriesNoCache(const LinearSeries &LS);

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

  /// @brief Erase ripple function-specialization related metadata that has been
  /// attached to the given function
  static void eraseFunctionSpecializationRelatedMetadata(Function &F);

  /// @brief Returns true if a function is marked for Ripple Specialization
  bool isPendingRippleSpecialization(const Function &F) const;

  /// @brief Returns true if a function is marked for Ripple Specialization
  bool shouldWaitForVoidReturnSpecialization(const Function &F) const;

  /// @brief Returns a unique name for a specialization of function name
  /// OriginalName
  static std::string getUniqueSpecializationName(StringRef OriginalName,
                                                 bool Final = false);

  /// @brief Returns the base (OriginalName) of the ripple function
  /// specialization, or an empty string if it is not a specialization.
  static StringRef getOriginalName(StringRef SpecializationName);

  /// @brief Gets the base function of this specialization, or nullptr if it is
  /// not a ripple specialization
  static Function *
  specializationOriginalFunction(const Function &Specialization);

  /// Store the name of the final specialization for void returning functions.
  /// That way we can call the final function w/o having to wait for the
  /// specialization to be processed by ripple.
  void registerFinalFunctionNameMetadata(Function *PendingSpecialization,
                                         const Function *FinalSpecialization);

  /// Retrieve the declaration of the processed specialization or nullptr if it
  /// has not been created yet
  Function *
  retrieveFinalSpecializationDecl(const Function *PendingSpecialization) const;

private:
  /// @brief Datatype returned when constructing linear series
  /// This also make sure that linear series are deleted when no more in use,
  /// allowing the creation of linear series for constants for example
  using ConstructedSeries = struct ConstructedSeries {
    /// @brief State of a ConstructedSeries
    enum CSState {
      /// @brief It's not a series!
      NotASeries = 0,
      /// @brief This is a finalized, valid, linear series
      ValidLinearSeries,
      /// @brief This is a temporary, non-useable, linear series
      PotentialLinearSeries,
    };
    // The LinearSeries itself
    LinearSeries *LS = nullptr;
    // The validity of this series
    CSState St = NotASeries;

    CSState getState() const { return St; }
    /// @brief Checks if this result is not a series
    bool isNotASeries() const { return getState() == NotASeries; }
    /// @brief Checks if this result is a useable series
    bool isValid() const { return getState() == ValidLinearSeries; }
    /// @brief Checks if this result is temporary and could yield a valid series
    /// later
    bool hasPotential() const { return getState() == PotentialLinearSeries; }
    /// @brief Checks if this result is a valid series
    operator bool() const { return isValid(); }

    ConstructedSeries() = default;
    ConstructedSeries(LinearSeries *LS, CSState St) : LS(LS), St(St) {
      if (LS)
        LS->Retain();
    }
    ConstructedSeries(
        std::initializer_list<std::variant<LinearSeries *, CSState>> List) {
      assert(List.size() == 0 || List.size() == 2);
      if (List.size() != 2)
        return;
      auto ListIt = List.begin();
      LS = std::get<LinearSeries *>(*ListIt);
      if (LS)
        LS->Retain();
      ++ListIt;
      St = std::get<CSState>(*ListIt);
    }
    ConstructedSeries(const ConstructedSeries &Other)
        : LS(Other.LS), St(Other.getState()) {
      if (LS)
        LS->Retain();
    }
    ConstructedSeries(ConstructedSeries &&Other)
        : LS(std::move(Other.LS)), St(std::move(Other.St)) {
      Other.LS = nullptr;
      Other.St = NotASeries;
    }
    ConstructedSeries &operator=(const ConstructedSeries &other) {
      if (LS)
        LS->Release();
      LS = other.LS;
      if (LS)
        LS->Retain();
      St = other.getState();
      return *this;
    }
    ~ConstructedSeries() {
      if (LS)
        LS->Release();
    }
  };
  using CSState = ConstructedSeries::CSState;
  friend raw_ostream &operator<<(raw_ostream &OS, const Ripple::CSState &State);
  friend raw_ostream &operator<<(raw_ostream &OS,
                                 const Ripple::ConstructedSeries &CS);

  /// @brief A cache structure used to cache constructed series
  using LSCache = struct {
    /// Fully valid LinearSeries
    DenseMap<AssertingVH<const Instruction>, LinearSeries *> Valid;
    /// Potential Series (due to PHI nodes)
    DenseMap<AssertingVH<const Instruction>, LinearSeries *> Potential;
    /// Linear Series that have been instantiated
    DenseMap<const LinearSeries *, AssertingVH<Value>> GeneratedSeries;
  };

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

  /// @brief A map from values to vector instructions
  DenseMap<PoisoningVH<Instruction>, AssertingVH<Value>>
      InstructionReplacementMapping;

  /// @brief Branch or switch instructions to vector conditions
  DenseMap<AssertingVH<Instruction>, AssertingVH<Value>> BranchAndSwitchVecCond;

  /// @brief A cache used by SimplifyQuery
  AssumptionAnalysis::Result &AssumptionCache;

  /// @brief Used to simplify expressions generated by LinearSeries w/ PHINodes
  SimplifyQuery SQ;

  /// @brief A convenience scalar shape
  const TensorShape ScalarShape;

  /// @brief A shape for values that should't be modified by Ripple, also has
  /// the nice property to be the neutral shape for broadcasting!
  const TensorShape &ShapeIgnoredByRipple = ScalarShape;

  /// @brief A cache of constructed linear series
  LSCache LsCache;

  /// @brief A cache of constructed slope instructions
  DenseSet<AssertingVH<Instruction>> SlopeInstructions;

  /// A n-dimensional load/store factory associated with the current module.
  NDLoadStoreFactory NdLoadStoreFac;

  DenseSet<AssertingVH<AllocaInst>> PromotableAlloca;
  DenseSet<AssertingVH<AllocaInst>> NonPromotableAlloca;

  /// @brief Argument shapes used for function specialization
  SmallVector<TensorShape, 8> ArgumentShapes;

  /// @brief A way to return the processing status of this function pass to the
  /// Ripple module pass
  ProcessingStatus &PS;

  /// @brief The pending and available specialization sets
  DenseSet<AssertingVH<Function>> &SpecializationsPending,
      &SpecializationsAvailable;

  /// @brief Combines two states for a binary operator
  static CSState combineStatesBinaryOp(CSState S, CSState S2);

  /// @brief Register a mapping between a scalar Instruction to a vector value
  /// This also sets the shape of the replacement
  void setReplacementFor(Instruction *ToReplace, Value *Replacement,
                         const TensorShape &Shape);

  /// @brief Returns \p U's tensor value and it's shape, i.e., the value can
  /// either be a "replacement Tensor value", a "LinearSeries instantiation" or
  /// the "original Use value".
  ///
  /// @pre Valid during/after genVectorInstructions and before ifConvert
  ///
  /// This method takes care of instantiating LinearSeries for users of LS.
  ///
  /// If both the Use and the User are LinearSeries with the same base shape
  /// returns the replacement value instead of the instantiation since we
  /// are building the replacement base and are not using the Series's value.
  std::pair<Value *, const TensorShape *> getTensorUse(const Use &U);

  /// DON'T USE THIS METHOD, use @ref getTensorUse(const Use&) instead!
  ///
  /// @brief Gets the "replacement" Tensor value of V and its shape.
  /// 1) The replacement of @p V if it exists
  /// 2) If @p V is a LinearSeries, returns the base replacement if it exists
  /// 3) None of 1) and 2) have replacement, returns @p V
  ///
  /// @warning If the return value is a Tensor AllocaInst, the shape should not
  /// be trusted because their shape depends on the User Instruction location.
  ///
  /// @pre Valid during/after genVectorInstructions and before ifConvert
  std::pair<Value *, const TensorShape *>
  replacementValueAndShape(Value *V) const;

  /// @brief Initializes *idRanks* with data extracted from ripple intrinsics in
  /// the function
  static DenseMap<PEIdentifier, DimSize>
  gatherRippleFunctionUsage(const Function &F);

  /// @brief The set of select instructions that requires masking when
  /// if-converting.
  /// For example multi-dimensional reductions are generated with a select as
  /// first instruction Select(true, vector, neutral_elem) so that we can mask
  /// them later.
  DenseSet<AssertingVH<SelectInst>> SelectToMaskWhenIfConvert;

  /// @brief The set of instructions that should not
  /// be masked when if-converting.
  DenseSet<AssertingVH<Instruction>> ToSkipMaskingWhenIfConvert;

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

  /// @brief Invalidates a tensor shape
  /// This is usually useful when deleting instructions
  void invalidateRippleDataFor(const Value *V);
  // Helper function to support DbgRecord and llvm.dbg intrinsics
  void invalidateRippleDataFor(const DbgRecord *Dbg) {}

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

  /// @brief Generates vector instructions for each vector Ripple shape
  void genVectorInstructions();

  /// @brief Post-process the function to remove scalar Instructions that have
  /// been vectorized.
  void vectorGenerationPostProcess();

  /// @brief Get the tensorUse of \p I's operands, indexed in range [StartIdx,
  /// EndIdx[, broadcast each of them to \p ToShape and return the resulting
  /// vector of broadcasted operands
  ///
  /// Functionally equivalent to:
  ///   map(getTensorUseAndBcast,
  ///       map(I->getOperandUse,
  ///           range(startIdx, EndIdx)
  ///       )
  ///   )
  ///
  /// @see getTensorUseAndBcast
  /// @pre Valid during/after genVectorInstructions and before ifConvert
  SmallVector<Value *, 8> tensorizedOperandsAndBroadcast(
      Instruction *I, const TensorShape &ToShape,
      unsigned StartIdx = std::numeric_limits<unsigned>().min(),
      unsigned EndIdx = std::numeric_limits<unsigned>().max());

  /// @brief Returns \p U value generated during Ripple's vector codegen
  /// phase being broadcasted \p ToShape.
  ///
  /// Functionally equivalent to (with automatic bcast error diagnostic
  /// reporting):
  ///   tensorBcast(getTensorUse(U), ToShape)
  ///
  /// @see getTensorUse
  /// @pre Valid during/after genVectorInstructions and before ifConvert
  Value *getTensorUseAndBcast(const Use &U, const TensorShape &ToShape);

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

  /// @brief If-conversion for vector branches
  /// Replaces the pattern "if x then a else b" into  mask(x, a) -> mask(!x, b)
  void ifConvert();

  /// @brief Apply mask to maskable operators in the given basic blocks
  /// @param BBs the basic blocks to consider
  /// @param VectorMask the mask value to apply
  /// @param MaskShape the mask tensor shape
  /// @param MaskInsertionPoint The location where mask reductions/broadcasts
  /// will be inserted
  void applyMaskToOps(ArrayRef<BasicBlock *> BBs, Value *VectorMask,
                      const TensorShape &MaskShape,
                      Instruction *MaskInsertionPoint);

  /// @brief Apply mask to maskable operators in the given basic block range
  /// @param BBs the basic block range to consider
  /// @param VectorMask the mask value to apply
  /// @param MaskShape the mask tensor shape
  /// @param MaskInsertionPoint The location where mask reductions/broadcasts
  /// will be inserted
  template <typename IteratorT>
  void applyMaskToOps(llvm::iterator_range<IteratorT> BBs, Value *VectorMask,
                      const TensorShape &MaskShape,
                      Instruction *MaskInsertionPoint);

  /// @brief Tests if all the reachable instruction of the function have an
  /// associated Ripple shape
  /// @return true if all instructions have a ripple shape, false otherwise
  bool allInstructionsHaveRippleShapes() const;

  /// @brief Simplifies the instructions in the function. This never creates new
  /// instructions (only constants). Returns true if simplifications were made.
  bool simplifyFunction();

  /// @brief Returns the set of basic blocks ]from, to] by doing a depth-first
  /// search starting at @p from and stopping when encountering @p to. @p to
  /// must post-dominate @p from.
  DenseSet<BasicBlock *> allBasicBlocksFromTo(BasicBlock *from,
                                              BasicBlock *to) const;

  /// @brief Build the masks for the lhs and rhs of the BranchInst
  void vectorBranchMasks(
      BranchInst *Branch, Value *VectorCondition,
      SmallVectorImpl<std::pair<BasicBlock *, Value *>> &TargetMasks,
      const TensorShape &MaskShape);

  /// @brief Build the masks for the switch branches
  /// The default condition is inserted last in TargetMasks
  void vectorSwitchMasks(
      SwitchInst *Switch, Value *VectorCondition,
      SmallVectorImpl<std::pair<BasicBlock *, Value *>> &TargetMasks,
      const TensorShape &MaskShape);

  /// @brief Depth first clones basic blocks starting from *start* only for
  /// BasicBlocks in BBs
  /// @param Start the first basic block to clone
  /// @param BBs the set of basic blocks to clone
  /// @param VMap the value map populated during the cloning
  /// @param ClonedBBs a vector to populate with pointers of cloned basic blocks
  void clonePathStartingWith(BasicBlock *Start,
                             const DenseSet<BasicBlock *> &BBs,
                             ValueToValueMapTy &VMap,
                             std::vector<BasicBlock *> &ClonedBBs);

  /// @brief Generates the instructions for a multi-dimensional reduction.
  /// The algorithm generates log2 shuffle and reduction instruction per
  /// dimension being reduced.
  /// @param reductionId the vp_reduction intrinsic ID
  /// @param vector the vector to reduce
  /// @param vectorShape the vector multi-dimensional shape
  /// @param reductionDimensions the dimensions indices that are reduced
  /// @return the output of the reduction
  Value *genMultiDimReduction(Intrinsic::ID reductionId, Value *vector,
                              const TensorShape &vectorShape,
                              const BitVector &reductionDimensions,
                              FMFSource FMFSource = {});

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

  /// @brief Apply a reduction followed by a broadcast to get a mask that can be
  /// applied to ExpectedShape.
  /// @param Mask The mask value
  /// @param MaskShape The mask shape
  /// @param ExpectedShape The expected mask shape
  Value *reduceBcastMaskToShape(Value *Mask, const TensorShape &MaskShape,
                                const TensorShape &ExpectedShape);

  /// @brief Returns true if the value has a valid type and shape for a linear
  /// series, i.e., integer or pointers and FromShape can be broadcasted to
  /// ToShape
  static bool canConstructSplatSeries(Value *V, const TensorShape &FromShape,
                                      const TensorShape &ToShape);

  /// @brief Common constructor to construct a series that splats V
  ConstructedSeries getSplatSeries(Value *V, const TensorShape &FromShape,
                                   const TensorShape &ToShape);

  /// @brief Constructs a linear series for a constant
  ConstructedSeries getLinearSeriesFor(Constant *C,
                                       const TensorShape &FromShape,
                                       const TensorShape &ToShape);

  /// @brief Constructs a linear series for an argument
  ConstructedSeries getLinearSeriesFor(Argument *A,
                                       const TensorShape &FromShape,
                                       const TensorShape &ToShape);

  /// @brief Constructs a linear series for an instruction
  ConstructedSeries getLinearSeriesFor(Instruction *I);

  /// @brief Try to promote a linear series. Returns a valid series if it could
  /// be promoted or a non-series otherwise.
  ConstructedSeries tryToPromoteLinearSeries(LinearSeries *LS);

  /// @brief Create the Value representing *this* linear series
  /// Prefer using instantiateCachedSeries instead of this function
  Value *instantiateLinearSeries(const LinearSeries *LS,
                                 bool UseLSCache = true);

  /// @brief Instantiate the linear series represented by CS after the
  /// instuction AfterI
  Value *instantiateCachedSeries(ConstructedSeries &CS, Instruction *AfterI);

  /// @brief Returns a caches LinearSeries for the instruction I
  ConstructedSeries getCachedSeries(const Instruction *I) const;

  /// @brief Simplify slopes which are empty with the value zero.
  /// @param Cache the cache of linear series
  void simplifySlopes();

  /// @brief Builds the value representing the slope of this LinearSeries
  Value *buildLinearSeriesSlope(const LinearSeries *LS);

  /// @brief Returns the cached linear series instantiation for the instruction
  /// I or nullptr if not a linear series or not instantiated
  Value *getCachedInstantiationFor(const Instruction *I) const;

  /// Clears a specified Instruction's valid linear series
  void clearValidSerie(const Instruction *I);

  /// @brief Clears potential linear series
  void clearPotentialSeries();

  /// @brief Checks that the parents of given linear series are valid linear
  /// series
  bool hasValidLinearSeriesRoots(LinearSeries *LS) const;

  /// @brief Clear all linear series caches
  void clearLinearSeriesCache();

  /// @brief Returns true if there are no vector dimensions, false otherwise
  bool hasNoVectorDimension() const;

  /// @brief Checks for any semantics issue with the use of ripple intrinsics or
  /// vectorization (if-conversion, impossible vector instruction generation,
  /// etc)
  Error checkRippleSemantics();

  /// @brief Checks the given ripple block intrinsics for issues
  Error checkRippleBlockIntrinsics(IntrinsicInst *I);

  /// @brief Checks the given reduction intrinsics for issues
  Error checkRippleReductionIntrinsics(IntrinsicInst *I);

  /// @brief Checks the given shuffle intrinsics for issues
  Error checkRippleShuffleIntrinsics(IntrinsicInst *I);

  /// @brief Checks the given rot-to-lower intrinsics for issues
  Error checkRippleRotToLowerIntrinsics(IntrinsicInst *I);

  /// @brief Checks that vector branch apply to a SESE region
  Error checkVectorBranch(Instruction *BranchOrSwitch);

  /// @brief Checks that creating a vector type for I is valid
  Error checkTypeCanBeVectorized(const Instruction *I);

  /// @brief Checks that vector store are valid
  Error checkRippleStore(const StoreInst *I) const;

  /// @brief Checks the function's return
  Error checkRippleFunctionReturn(const ReturnInst *Return) const;

  /// @brief Creates a DILocalVariable of vector type from a scalar
  /// DILocalVariable, or nullptr if the DILocalVariable has no type
  DILocalVariable *createVectorLocalFromScalarLocal(
      DIBuilder &DIB, DILocalVariable *LocalVariable, const TensorShape &Shape);

  /// @brief Fixes Variable Debug intrinsics (llvm.dbg) or records with vector
  /// types if we replaced the value being targeted by the debug info.
  template <typename T>
  void processDebugIntrinsicOrRecord(DIBuilder &DIB, T &DbgMetadata);

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

  /// @brief Returns true if this instruction requires masking during the
  /// if-conversion process, false otherwise
  bool maskInstructionWhenIfConvert(const Instruction *I) const;

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

  /// @brief Returns true if the function has shape information embedded
  static bool hasRippleShapeMetadata(const Function &F);

  /// @brief Set the function shape metadata
  void setFunctionShapeMetadata(Function &F,
                                ArrayRef<const TensorShape *> ArgShapes,
                                const TensorShape &ReturnShape);

  /// @brief Retrieve the tensor shapes of the function's argument and return
  /// when it's a specialization.
  bool getFunctionShapeMetadata(
      const Function &F,
      SmallVectorImpl<std::unique_ptr<TensorShape>> &ArgShapes,
      std::unique_ptr<TensorShape> &ReturnShape) const;

  /// @brief Returns true if a function is safe to be specialized by Ripple
  static bool canBeSpecialized(const Function *F);

  /// @brief Returns the specialization for Call if there is one in the module
  /// and the return tensor shape.
  /// The returned function may be a call that has not yet been processed by
  /// ripple and hence the return shape cannot be trusted. The caller can test
  /// that by using @ref isPendingRippleSpecialization and may have to postpone
  /// processing until the function has been processed.
  std::pair<Function *, TensorShape>
  getRippleSpecializationFor(const CallInst &Call) const;

  /// @brief Initiate the process of specializing the given call
  bool requestSpecializationFor(const CallInst &Call);

  /// @brief Called when ripple is done processing a specialization, to fix the
  /// function prototype now that all the types have settled.
  /// @pre isPendingRippleSpecialization is true
  void finishSpecialization();

  /// @brief Request the masked specialization of a given non-masked
  /// specialization
  /// @param Specialization The specialization function
  /// @return The masked specialization, or nullptr if it does not exist
  Function *getMaskedSpecialization(const Function &Specialization) const;

  /// @brief Returns the name of the masked specialization given a
  /// specialization function
  std::string getMaskedSpecializationName(const Function &Specialization) const;

  /// @brief Returns the element type of a specialization mask
  IntegerType *getSpecializationMaskElementType();

  /// @brief Given a function's operand value range, returns the shape of the
  /// specialization's mask
  /// This function queries the shape of the operands to infer the shape of the
  /// mask.
  template <typename IteratorT>
  Expected<TensorShape>
  getSpecializationMaskShape(llvm::iterator_range<IteratorT> Args) const;

  /// @brief Returns the tensor shape of RippleSetShape
  TensorShape setShapeToTensorShape(const IntrinsicInst *RippleSetShape) const;

  /// @brief Returns the size of the dimension requested by a the
  /// ripple.get.size intrinsic
  TensorShape::DimSize
  getRippleGetSizeValue(const IntrinsicInst *RippleGetSize);

  /// @brief Populates @p BSArgs with pairs <argument index,
  /// ripple_set_block_shape intrinsic> for each argument we can trace back to
  /// block set shape or use the block shape with a ripple intrinsic
  void specializationCallBlockShapeArgs(
      const CallInst *CI,
      SmallVectorImpl<std::pair<unsigned, IntrinsicInst *>> &BSArgs);

  /// @brief Checks that all the ripple instrinsics in this function that rely
  /// on a block size can find them and returns an error otherwise.
  Error checkBlockShapeUsage(const Function &F);

  /// @brief Ripple vectorizes calls when they have at least one tensor argument
  /// with vector shapes and no argument with regular (LLVM) vector type
  /// arguments.
  bool rippleVectorizeCall(const CallInst &CI) const;
};

inline raw_ostream &operator<<(raw_ostream &OS, const Ripple::CSState &State) {
  switch (State) {
  case Ripple::CSState::NotASeries:
    OS << "Not a series";
    break;
  case Ripple::CSState::PotentialLinearSeries:
    OS << "Potential series";
    break;
  case Ripple::CSState::ValidLinearSeries:
    OS << "Valid series";
    break;
  }
  return OS;
}

inline raw_ostream &operator<<(raw_ostream &OS, const LinearSeries &LS) {
  LS.print(OS);
  return OS;
}

inline raw_ostream &operator<<(raw_ostream &OS,
                               const Ripple::ConstructedSeries &CS) {
  OS << "(" << CS.getState() << ")";
  if (!CS.isNotASeries())
    OS << *CS.LS;
  return OS;
}

/**
 * @brief Returns `true` iff BranchingBB has exactly size two (immediate)
 * successors, and, exactly one of them exhibits a cycle through `BranchingBB`
 * that does not go through PDom.
 *
 * @param BranchingBB The Basic block with a branc which is to be checked for a
 * trivial loop-like back-edge.
 * @param PDom Immediate post-dominator of BranchingBB.
 * @param Targets All Targets of BranchingBB.
 */
bool hasTrivialLoopLikeBackEdge(BasicBlock *BranchingBB, BasicBlock *PDom,
                                DominatorTreeAnalysis::Result &DT);

} // namespace llvm

#endif // LLVM_TRANSFORMS_VECTORIZE_RIPPLE_H
