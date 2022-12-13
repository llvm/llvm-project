//===- SparseTensorRuntime.cpp - SparseTensor runtime support lib ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a light-weight runtime support library for
// manipulating sparse tensors from MLIR.  More specifically, it provides
// C-API wrappers so that MLIR-generated code can call into the C++ runtime
// support library.  The functionality provided in this library is meant
// to simplify benchmarking, testing, and debugging of MLIR code operating
// on sparse tensors.  However, the provided functionality is **not**
// part of core MLIR itself.
//
// The following memory-resident sparse storage schemes are supported:
//
// (a) A coordinate scheme for temporarily storing and lexicographically
//     sorting a sparse tensor by index (SparseTensorCOO).
//
// (b) A "one-size-fits-all" sparse tensor storage scheme defined by
//     per-dimension sparse/dense annnotations together with a dimension
//     ordering used by MLIR compiler-generated code (SparseTensorStorage).
//
// The following external formats are supported:
//
// (1) Matrix Market Exchange (MME): *.mtx
//     https://math.nist.gov/MatrixMarket/formats.html
//
// (2) Formidable Repository of Open Sparse Tensors and Tools (FROSTT): *.tns
//     http://frostt.io/tensors/file-formats.html
//
// Two public APIs are supported:
//
// (I) Methods operating on MLIR buffers (memrefs) to interact with sparse
//     tensors. These methods should be used exclusively by MLIR
//     compiler-generated code.
//
// (II) Methods that accept C-style data structures to interact with sparse
//      tensors. These methods can be used by any external runtime that wants
//      to interact with MLIR compiler-generated code.
//
// In both cases (I) and (II), the SparseTensorStorage format is externally
// only visible as an opaque pointer.
//
//===----------------------------------------------------------------------===//

#include "mlir/ExecutionEngine/SparseTensorRuntime.h"

#ifdef MLIR_CRUNNERUTILS_DEFINE_FUNCTIONS

#include "mlir/ExecutionEngine/SparseTensor/ArithmeticUtils.h"
#include "mlir/ExecutionEngine/SparseTensor/COO.h"
#include "mlir/ExecutionEngine/SparseTensor/ErrorHandling.h"
#include "mlir/ExecutionEngine/SparseTensor/File.h"
#include "mlir/ExecutionEngine/SparseTensor/PermutationRef.h"
#include "mlir/ExecutionEngine/SparseTensor/Storage.h"

#include <cstring>
#include <numeric>

using namespace mlir::sparse_tensor;

//===----------------------------------------------------------------------===//
//
// Implementation details for public functions, which don't have a good
// place to live in the C++ library this file is wrapping.
//
//===----------------------------------------------------------------------===//

namespace {

/// Wrapper class to avoid memory leakage issues.  The `SparseTensorCOO<V>`
/// class provides a standard C++ iterator interface, where the iterator
/// is implemented as per `std::vector`'s iterator.  However, for MLIR's
/// usage we need to have an iterator which also holds onto the underlying
/// `SparseTensorCOO<V>` so that it can be freed whenever the iterator
/// is freed.
//
// We name this `SparseTensorIterator` rather than `SparseTensorCOOIterator`
// for future-proofing, since the use of `SparseTensorCOO` is an
// implementation detail that we eventually want to change (e.g., to
// use `SparseTensorEnumerator` directly, rather than constructing the
// intermediate `SparseTensorCOO` at all).
template <typename V>
class SparseTensorIterator final {
public:
  /// This ctor requires `coo` to be a non-null pointer to a dynamically
  /// allocated object, and takes ownership of that object.  Therefore,
  /// callers must not free the underlying COO object, since the iterator's
  /// dtor will do so.
  explicit SparseTensorIterator(const SparseTensorCOO<V> *coo)
      : coo(coo), it(coo->begin()), end(coo->end()) {}

  ~SparseTensorIterator() { delete coo; }

  // Disable copy-ctor and copy-assignment, to prevent double-free.
  SparseTensorIterator(const SparseTensorIterator<V> &) = delete;
  SparseTensorIterator<V> &operator=(const SparseTensorIterator<V> &) = delete;

  /// Gets the next element.  If there are no remaining elements, then
  /// returns nullptr.
  const Element<V> *getNext() { return it < end ? &*it++ : nullptr; }

private:
  const SparseTensorCOO<V> *const coo; // Owning pointer.
  typename SparseTensorCOO<V>::const_iterator it;
  const typename SparseTensorCOO<V>::const_iterator end;
};

// TODO: When using this library from MLIR, the `toMLIRSparseTensor`/
// `IMPL_CONVERTTOMLIRSPARSETENSOR` and `fromMLIRSparseTensor`/
// `IMPL_CONVERTFROMMLIRSPARSETENSOR` constructs will be codegened away;
// therefore, these functions are only used by PyTACO, one place in the
// Python integration tests, and possibly by out-of-tree projects.
// This is notable because neither function can be easily generalized
// to handle non-permutations.  In particular, while we could adjust
// the functions to take all the arguments they'd need, that would just
// push the problem into client code.  So if we want to generalize these
// functions to support non-permutations, we'll need to figure out how
// to do so without putting undue burden on clients.

/// Initializes sparse tensor from an external COO-flavored format.
/// The `rank` argument is both dimension-rank and level-rank, and the
/// `dim2lvl` argument must be a permutation.
/// Used by `IMPL_CONVERTTOMLIRSPARSETENSOR`.
//
// TODO: generalize beyond 64-bit indices.
template <typename V>
static SparseTensorStorage<uint64_t, uint64_t, V> *
toMLIRSparseTensor(uint64_t rank, uint64_t nse, const uint64_t *dimSizes,
                   const V *values, const uint64_t *dimIndices,
                   const uint64_t *dim2lvl, const DimLevelType *lvlTypes) {
#ifndef NDEBUG
  // Verify that the sparsity values are supported.
  // TODO: update this check to match what we actually support.
  for (uint64_t i = 0; i < rank; ++i)
    if (lvlTypes[i] != DimLevelType::Dense &&
        lvlTypes[i] != DimLevelType::Compressed)
      MLIR_SPARSETENSOR_FATAL("unsupported level type: %d\n",
                              static_cast<uint8_t>(lvlTypes[i]));
#endif
  // Verify that `dim2lvl` is a permutation of `[0..(rank-1)]`.
  // NOTE: The construction of `lvlSizes` and `lvl2dim` don't generalize
  // to arbitrary `dim2lvl` mappings.  Whereas constructing `lvlInd` from
  // `dimInd` does (though the details would have to be updated, just
  // like for `IMPL_ADDELT`).
  detail::PermutationRef d2l(rank, dim2lvl);
  // Convert external format to internal COO.
  auto lvlSizes = d2l.pushforward(rank, dimSizes);
  auto *lvlCOO = new SparseTensorCOO<V>(lvlSizes, nse);
  std::vector<uint64_t> lvlInd(rank);
  const uint64_t *dimInd = dimIndices;
  for (uint64_t i = 0; i < nse; ++i) {
    d2l.pushforward(rank, dimInd, lvlInd.data());
    lvlCOO->add(lvlInd, values[i]);
    dimInd += rank;
  }
  // Return sparse tensor storage format as opaque pointer.
  auto lvl2dim = d2l.inverse();
  auto *tensor = SparseTensorStorage<uint64_t, uint64_t, V>::newFromCOO(
      rank, dimSizes, rank, lvlTypes, lvl2dim.data(), *lvlCOO);
  delete lvlCOO;
  return tensor;
}

/// Converts a sparse tensor to an external COO-flavored format.
/// Used by `IMPL_CONVERTFROMMLIRSPARSETENSOR`.
//
// TODO: Currently, values are copied from SparseTensorStorage to
// SparseTensorCOO, then to the output.  We may want to reduce the number
// of copies.
//
// TODO: generalize beyond 64-bit indices, no dim ordering, all dimensions
// compressed
template <typename V>
static void
fromMLIRSparseTensor(const SparseTensorStorage<uint64_t, uint64_t, V> *tensor,
                     uint64_t *pRank, uint64_t *pNse, uint64_t **pShape,
                     V **pValues, uint64_t **pIndices) {
  assert(tensor && "Received nullptr for tensor");
  uint64_t dimRank = tensor->getDimRank();
  const auto &dimSizes = tensor->getDimSizes();
  std::vector<uint64_t> identityPerm(dimRank);
  std::iota(identityPerm.begin(), identityPerm.end(), 0);
  SparseTensorCOO<V> *coo =
      tensor->toCOO(dimRank, dimSizes.data(), dimRank, identityPerm.data());

  const std::vector<Element<V>> &elements = coo->getElements();
  uint64_t nse = elements.size();

  const auto &cooSizes = coo->getDimSizes();
  assert(cooSizes.size() == dimRank && "Rank mismatch");
  uint64_t *shape = new uint64_t[dimRank];
  std::memcpy((void *)shape, (const void *)cooSizes.data(),
              sizeof(uint64_t) * dimRank);

  V *values = new V[nse];
  uint64_t *indices = new uint64_t[dimRank * nse];

  for (uint64_t i = 0, base = 0; i < nse; ++i) {
    values[i] = elements[i].value;
    for (uint64_t d = 0; d < dimRank; ++d)
      indices[base + d] = elements[i].indices[d];
    base += dimRank;
  }

  delete coo;
  *pRank = dimRank;
  *pNse = nse;
  *pShape = shape;
  *pValues = values;
  *pIndices = indices;
}

//===----------------------------------------------------------------------===//
//
// Utilities for manipulating `StridedMemRefType`.
//
//===----------------------------------------------------------------------===//

// We shouldn't need to use `detail::safelyEQ` here since the `1` is a literal.
#define ASSERT_NO_STRIDE(MEMREF)                                               \
  do {                                                                         \
    assert((MEMREF) && "Memref is nullptr");                                   \
    assert(((MEMREF)->strides[0] == 1) && "Memref has non-trivial stride");    \
  } while (false)

// All our functions use `uint64_t` for ranks, but `StridedMemRefType::sizes`
// uses `int64_t` on some platforms.  So we explicitly cast this lookup to
// ensure we get a consistent type, and we use `checkOverflowCast` rather
// than `static_cast` just to be extremely sure that the casting can't
// go awry.  (The cast should aways be safe since (1) sizes should never
// be negative, and (2) the maximum `int64_t` is smaller than the maximum
// `uint64_t`.  But it's better to be safe than sorry.)
#define MEMREF_GET_USIZE(MEMREF)                                               \
  detail::checkOverflowCast<uint64_t>((MEMREF)->sizes[0])

#define ASSERT_USIZE_EQ(MEMREF, SZ)                                            \
  assert(detail::safelyEQ(MEMREF_GET_USIZE(MEMREF), (SZ)) &&                   \
         "Memref size mismatch")

#define MEMREF_GET_PAYLOAD(MEMREF) ((MEMREF)->data + (MEMREF)->offset)

/// Initializes the memref with the provided size and data pointer.  This
/// is designed for functions which want to "return" a memref that aliases
/// into memory owned by some other object (e.g., `SparseTensorStorage`),
/// without doing any actual copying.  (The "return" is in scarequotes
/// because the `_mlir_ciface_` calling convention migrates any returned
/// memrefs into an out-parameter passed before all the other function
/// parameters.)
///
/// We make this a function rather than a macro mainly for type safety
/// reasons.  This function does not modify the data pointer, but it
/// cannot be marked `const` because it is stored into the (necessarily)
/// non-`const` memref.  This function is templated over the `DataSizeT`
/// to work around signedness warnings due to many data types having
/// varying signedness across different platforms.  The templating allows
/// this function to ensure that it does the right thing and never
/// introduces errors due to implicit conversions.
template <typename DataSizeT, typename T>
static inline void aliasIntoMemref(DataSizeT size, T *data,
                                   StridedMemRefType<T, 1> &ref) {
  ref.basePtr = ref.data = data;
  ref.offset = 0;
  using MemrefSizeT = typename std::remove_reference_t<decltype(ref.sizes[0])>;
  ref.sizes[0] = detail::checkOverflowCast<MemrefSizeT>(size);
  ref.strides[0] = 1;
}

} // anonymous namespace

extern "C" {

//===----------------------------------------------------------------------===//
//
// Public functions which operate on MLIR buffers (memrefs) to interact
// with sparse tensors (which are only visible as opaque pointers externally).
//
//===----------------------------------------------------------------------===//

#define CASE(p, i, v, P, I, V)                                                 \
  if (ptrTp == (p) && indTp == (i) && valTp == (v)) {                          \
    switch (action) {                                                          \
    case Action::kEmpty:                                                       \
      return SparseTensorStorage<P, I, V>::newEmpty(                           \
          dimRank, dimSizes, lvlRank, lvlSizes, lvlTypes, lvl2dim);            \
    case Action::kFromCOO: {                                                   \
      assert(ptr && "Received nullptr for SparseTensorCOO object");            \
      auto &coo = *static_cast<SparseTensorCOO<V> *>(ptr);                     \
      return SparseTensorStorage<P, I, V>::newFromCOO(                         \
          dimRank, dimSizes, lvlRank, lvlTypes, lvl2dim, coo);                 \
    }                                                                          \
    case Action::kSparseToSparse: {                                            \
      assert(ptr && "Received nullptr for SparseTensorStorage object");        \
      auto &tensor = *static_cast<SparseTensorStorageBase *>(ptr);             \
      return SparseTensorStorage<P, I, V>::newFromSparseTensor(                \
          dimRank, dimSizes, lvlRank, lvlSizes, lvlTypes, lvl2dim, dimRank,    \
          dim2lvl, tensor);                                                    \
    }                                                                          \
    case Action::kEmptyCOO:                                                    \
      return new SparseTensorCOO<V>(lvlRank, lvlSizes);                        \
    case Action::kToCOO: {                                                     \
      assert(ptr && "Received nullptr for SparseTensorStorage object");        \
      auto &tensor = *static_cast<SparseTensorStorage<P, I, V> *>(ptr);        \
      return tensor.toCOO(lvlRank, lvlSizes, dimRank, dim2lvl);                \
    }                                                                          \
    case Action::kToIterator: {                                                \
      assert(ptr && "Received nullptr for SparseTensorStorage object");        \
      auto &tensor = *static_cast<SparseTensorStorage<P, I, V> *>(ptr);        \
      auto *coo = tensor.toCOO(lvlRank, lvlSizes, dimRank, dim2lvl);           \
      return new SparseTensorIterator<V>(coo);                                 \
    }                                                                          \
    }                                                                          \
    MLIR_SPARSETENSOR_FATAL("unknown action: %d\n",                            \
                            static_cast<uint32_t>(action));                    \
  }

#define CASE_SECSAME(p, v, P, V) CASE(p, p, v, P, P, V)

// Assume index_type is in fact uint64_t, so that _mlir_ciface_newSparseTensor
// can safely rewrite kIndex to kU64.  We make this assertion to guarantee
// that this file cannot get out of sync with its header.
static_assert(std::is_same<index_type, uint64_t>::value,
              "Expected index_type == uint64_t");

// TODO: this swiss-army-knife should be split up into separate functions
// for each action, since the various actions don't agree on (1) whether
// the first two arguments are "sizes" vs "shapes", (2) whether the "lvl"
// arguments are actually storage-levels vs target tensor-dimensions,
// (3) whether all the arguments are actually used/required.
void *_mlir_ciface_newSparseTensor( // NOLINT
    StridedMemRefType<index_type, 1> *dimSizesRef,
    StridedMemRefType<index_type, 1> *lvlSizesRef,
    StridedMemRefType<DimLevelType, 1> *lvlTypesRef,
    StridedMemRefType<index_type, 1> *lvl2dimRef,
    StridedMemRefType<index_type, 1> *dim2lvlRef, OverheadType ptrTp,
    OverheadType indTp, PrimaryType valTp, Action action, void *ptr) {
  ASSERT_NO_STRIDE(dimSizesRef);
  ASSERT_NO_STRIDE(lvlSizesRef);
  ASSERT_NO_STRIDE(lvlTypesRef);
  ASSERT_NO_STRIDE(lvl2dimRef);
  ASSERT_NO_STRIDE(dim2lvlRef);
  const uint64_t dimRank = MEMREF_GET_USIZE(dimSizesRef);
  const uint64_t lvlRank = MEMREF_GET_USIZE(lvlSizesRef);
  ASSERT_USIZE_EQ(dim2lvlRef, dimRank);
  ASSERT_USIZE_EQ(lvlTypesRef, lvlRank);
  ASSERT_USIZE_EQ(lvl2dimRef, lvlRank);
  const index_type *dimSizes = MEMREF_GET_PAYLOAD(dimSizesRef);
  const index_type *lvlSizes = MEMREF_GET_PAYLOAD(lvlSizesRef);
  const DimLevelType *lvlTypes = MEMREF_GET_PAYLOAD(lvlTypesRef);
  const index_type *lvl2dim = MEMREF_GET_PAYLOAD(lvl2dimRef);
  const index_type *dim2lvl = MEMREF_GET_PAYLOAD(dim2lvlRef);

  // Rewrite kIndex to kU64, to avoid introducing a bunch of new cases.
  // This is safe because of the static_assert above.
  if (ptrTp == OverheadType::kIndex)
    ptrTp = OverheadType::kU64;
  if (indTp == OverheadType::kIndex)
    indTp = OverheadType::kU64;

  // Double matrices with all combinations of overhead storage.
  CASE(OverheadType::kU64, OverheadType::kU64, PrimaryType::kF64, uint64_t,
       uint64_t, double);
  CASE(OverheadType::kU64, OverheadType::kU32, PrimaryType::kF64, uint64_t,
       uint32_t, double);
  CASE(OverheadType::kU64, OverheadType::kU16, PrimaryType::kF64, uint64_t,
       uint16_t, double);
  CASE(OverheadType::kU64, OverheadType::kU8, PrimaryType::kF64, uint64_t,
       uint8_t, double);
  CASE(OverheadType::kU32, OverheadType::kU64, PrimaryType::kF64, uint32_t,
       uint64_t, double);
  CASE(OverheadType::kU32, OverheadType::kU32, PrimaryType::kF64, uint32_t,
       uint32_t, double);
  CASE(OverheadType::kU32, OverheadType::kU16, PrimaryType::kF64, uint32_t,
       uint16_t, double);
  CASE(OverheadType::kU32, OverheadType::kU8, PrimaryType::kF64, uint32_t,
       uint8_t, double);
  CASE(OverheadType::kU16, OverheadType::kU64, PrimaryType::kF64, uint16_t,
       uint64_t, double);
  CASE(OverheadType::kU16, OverheadType::kU32, PrimaryType::kF64, uint16_t,
       uint32_t, double);
  CASE(OverheadType::kU16, OverheadType::kU16, PrimaryType::kF64, uint16_t,
       uint16_t, double);
  CASE(OverheadType::kU16, OverheadType::kU8, PrimaryType::kF64, uint16_t,
       uint8_t, double);
  CASE(OverheadType::kU8, OverheadType::kU64, PrimaryType::kF64, uint8_t,
       uint64_t, double);
  CASE(OverheadType::kU8, OverheadType::kU32, PrimaryType::kF64, uint8_t,
       uint32_t, double);
  CASE(OverheadType::kU8, OverheadType::kU16, PrimaryType::kF64, uint8_t,
       uint16_t, double);
  CASE(OverheadType::kU8, OverheadType::kU8, PrimaryType::kF64, uint8_t,
       uint8_t, double);

  // Float matrices with all combinations of overhead storage.
  CASE(OverheadType::kU64, OverheadType::kU64, PrimaryType::kF32, uint64_t,
       uint64_t, float);
  CASE(OverheadType::kU64, OverheadType::kU32, PrimaryType::kF32, uint64_t,
       uint32_t, float);
  CASE(OverheadType::kU64, OverheadType::kU16, PrimaryType::kF32, uint64_t,
       uint16_t, float);
  CASE(OverheadType::kU64, OverheadType::kU8, PrimaryType::kF32, uint64_t,
       uint8_t, float);
  CASE(OverheadType::kU32, OverheadType::kU64, PrimaryType::kF32, uint32_t,
       uint64_t, float);
  CASE(OverheadType::kU32, OverheadType::kU32, PrimaryType::kF32, uint32_t,
       uint32_t, float);
  CASE(OverheadType::kU32, OverheadType::kU16, PrimaryType::kF32, uint32_t,
       uint16_t, float);
  CASE(OverheadType::kU32, OverheadType::kU8, PrimaryType::kF32, uint32_t,
       uint8_t, float);
  CASE(OverheadType::kU16, OverheadType::kU64, PrimaryType::kF32, uint16_t,
       uint64_t, float);
  CASE(OverheadType::kU16, OverheadType::kU32, PrimaryType::kF32, uint16_t,
       uint32_t, float);
  CASE(OverheadType::kU16, OverheadType::kU16, PrimaryType::kF32, uint16_t,
       uint16_t, float);
  CASE(OverheadType::kU16, OverheadType::kU8, PrimaryType::kF32, uint16_t,
       uint8_t, float);
  CASE(OverheadType::kU8, OverheadType::kU64, PrimaryType::kF32, uint8_t,
       uint64_t, float);
  CASE(OverheadType::kU8, OverheadType::kU32, PrimaryType::kF32, uint8_t,
       uint32_t, float);
  CASE(OverheadType::kU8, OverheadType::kU16, PrimaryType::kF32, uint8_t,
       uint16_t, float);
  CASE(OverheadType::kU8, OverheadType::kU8, PrimaryType::kF32, uint8_t,
       uint8_t, float);

  // Two-byte floats with both overheads of the same type.
  CASE_SECSAME(OverheadType::kU64, PrimaryType::kF16, uint64_t, f16);
  CASE_SECSAME(OverheadType::kU64, PrimaryType::kBF16, uint64_t, bf16);
  CASE_SECSAME(OverheadType::kU32, PrimaryType::kF16, uint32_t, f16);
  CASE_SECSAME(OverheadType::kU32, PrimaryType::kBF16, uint32_t, bf16);
  CASE_SECSAME(OverheadType::kU16, PrimaryType::kF16, uint16_t, f16);
  CASE_SECSAME(OverheadType::kU16, PrimaryType::kBF16, uint16_t, bf16);
  CASE_SECSAME(OverheadType::kU8, PrimaryType::kF16, uint8_t, f16);
  CASE_SECSAME(OverheadType::kU8, PrimaryType::kBF16, uint8_t, bf16);

  // Integral matrices with both overheads of the same type.
  CASE_SECSAME(OverheadType::kU64, PrimaryType::kI64, uint64_t, int64_t);
  CASE_SECSAME(OverheadType::kU64, PrimaryType::kI32, uint64_t, int32_t);
  CASE_SECSAME(OverheadType::kU64, PrimaryType::kI16, uint64_t, int16_t);
  CASE_SECSAME(OverheadType::kU64, PrimaryType::kI8, uint64_t, int8_t);
  CASE_SECSAME(OverheadType::kU32, PrimaryType::kI64, uint32_t, int64_t);
  CASE_SECSAME(OverheadType::kU32, PrimaryType::kI32, uint32_t, int32_t);
  CASE_SECSAME(OverheadType::kU32, PrimaryType::kI16, uint32_t, int16_t);
  CASE_SECSAME(OverheadType::kU32, PrimaryType::kI8, uint32_t, int8_t);
  CASE_SECSAME(OverheadType::kU16, PrimaryType::kI64, uint16_t, int64_t);
  CASE_SECSAME(OverheadType::kU16, PrimaryType::kI32, uint16_t, int32_t);
  CASE_SECSAME(OverheadType::kU16, PrimaryType::kI16, uint16_t, int16_t);
  CASE_SECSAME(OverheadType::kU16, PrimaryType::kI8, uint16_t, int8_t);
  CASE_SECSAME(OverheadType::kU8, PrimaryType::kI64, uint8_t, int64_t);
  CASE_SECSAME(OverheadType::kU8, PrimaryType::kI32, uint8_t, int32_t);
  CASE_SECSAME(OverheadType::kU8, PrimaryType::kI16, uint8_t, int16_t);
  CASE_SECSAME(OverheadType::kU8, PrimaryType::kI8, uint8_t, int8_t);

  // Complex matrices with wide overhead.
  CASE_SECSAME(OverheadType::kU64, PrimaryType::kC64, uint64_t, complex64);
  CASE_SECSAME(OverheadType::kU64, PrimaryType::kC32, uint64_t, complex32);

  // Unsupported case (add above if needed).
  // TODO: better pretty-printing of enum values!
  MLIR_SPARSETENSOR_FATAL(
      "unsupported combination of types: <P=%d, I=%d, V=%d>\n",
      static_cast<int>(ptrTp), static_cast<int>(indTp),
      static_cast<int>(valTp));
}
#undef CASE
#undef CASE_SECSAME

#define IMPL_SPARSEVALUES(VNAME, V)                                            \
  void _mlir_ciface_sparseValues##VNAME(StridedMemRefType<V, 1> *ref,          \
                                        void *tensor) {                        \
    assert(ref &&tensor);                                                      \
    std::vector<V> *v;                                                         \
    static_cast<SparseTensorStorageBase *>(tensor)->getValues(&v);             \
    assert(v);                                                                 \
    aliasIntoMemref(v->size(), v->data(), *ref);                               \
  }
MLIR_SPARSETENSOR_FOREVERY_V(IMPL_SPARSEVALUES)
#undef IMPL_SPARSEVALUES

#define IMPL_GETOVERHEAD(NAME, TYPE, LIB)                                      \
  void _mlir_ciface_##NAME(StridedMemRefType<TYPE, 1> *ref, void *tensor,      \
                           index_type d) {                                     \
    assert(ref &&tensor);                                                      \
    std::vector<TYPE> *v;                                                      \
    static_cast<SparseTensorStorageBase *>(tensor)->LIB(&v, d);                \
    assert(v);                                                                 \
    aliasIntoMemref(v->size(), v->data(), *ref);                               \
  }
#define IMPL_SPARSEPOINTERS(PNAME, P)                                          \
  IMPL_GETOVERHEAD(sparsePointers##PNAME, P, getPointers)
MLIR_SPARSETENSOR_FOREVERY_O(IMPL_SPARSEPOINTERS)
#undef IMPL_SPARSEPOINTERS

#define IMPL_SPARSEINDICES(INAME, I)                                           \
  IMPL_GETOVERHEAD(sparseIndices##INAME, I, getIndices)
MLIR_SPARSETENSOR_FOREVERY_O(IMPL_SPARSEINDICES)
#undef IMPL_SPARSEINDICES
#undef IMPL_GETOVERHEAD

// TODO: while this API design will work for arbitrary dim2lvl mappings,
// we should probably move the `dimInd`-to-`lvlInd` computation into codegen
// (since that could enable optimizations to remove the intermediate memref).
#define IMPL_ADDELT(VNAME, V)                                                  \
  void *_mlir_ciface_addElt##VNAME(                                            \
      void *lvlCOO, StridedMemRefType<V, 0> *vref,                             \
      StridedMemRefType<index_type, 1> *dimIndRef,                             \
      StridedMemRefType<index_type, 1> *dim2lvlRef) {                          \
    assert(lvlCOO &&vref);                                                     \
    ASSERT_NO_STRIDE(dimIndRef);                                               \
    ASSERT_NO_STRIDE(dim2lvlRef);                                              \
    const uint64_t rank = MEMREF_GET_USIZE(dimIndRef);                         \
    ASSERT_USIZE_EQ(dim2lvlRef, rank);                                         \
    const index_type *dimInd = MEMREF_GET_PAYLOAD(dimIndRef);                  \
    const index_type *dim2lvl = MEMREF_GET_PAYLOAD(dim2lvlRef);                \
    std::vector<index_type> lvlInd(rank);                                      \
    for (uint64_t d = 0; d < rank; ++d)                                        \
      lvlInd[dim2lvl[d]] = dimInd[d];                                          \
    V *value = MEMREF_GET_PAYLOAD(vref);                                       \
    static_cast<SparseTensorCOO<V> *>(lvlCOO)->add(lvlInd, *value);            \
    return lvlCOO;                                                             \
  }
MLIR_SPARSETENSOR_FOREVERY_V(IMPL_ADDELT)
#undef IMPL_ADDELT

#define IMPL_GETNEXT(VNAME, V)                                                 \
  bool _mlir_ciface_getNext##VNAME(void *iter,                                 \
                                   StridedMemRefType<index_type, 1> *iref,     \
                                   StridedMemRefType<V, 0> *vref) {            \
    assert(iter &&vref);                                                       \
    ASSERT_NO_STRIDE(iref);                                                    \
    index_type *indx = MEMREF_GET_PAYLOAD(iref);                               \
    V *value = MEMREF_GET_PAYLOAD(vref);                                       \
    const uint64_t isize = MEMREF_GET_USIZE(iref);                             \
    const Element<V> *elem =                                                   \
        static_cast<SparseTensorIterator<V> *>(iter)->getNext();               \
    if (elem == nullptr)                                                       \
      return false;                                                            \
    for (uint64_t r = 0; r < isize; r++)                                       \
      indx[r] = elem->indices[r];                                              \
    *value = elem->value;                                                      \
    return true;                                                               \
  }
MLIR_SPARSETENSOR_FOREVERY_V(IMPL_GETNEXT)
#undef IMPL_GETNEXT

#define IMPL_LEXINSERT(VNAME, V)                                               \
  void _mlir_ciface_lexInsert##VNAME(void *tensor,                             \
                                     StridedMemRefType<index_type, 1> *cref,   \
                                     StridedMemRefType<V, 0> *vref) {          \
    assert(tensor &&vref);                                                     \
    ASSERT_NO_STRIDE(cref);                                                    \
    index_type *cursor = MEMREF_GET_PAYLOAD(cref);                             \
    assert(cursor);                                                            \
    V *value = MEMREF_GET_PAYLOAD(vref);                                       \
    static_cast<SparseTensorStorageBase *>(tensor)->lexInsert(cursor, *value); \
  }
MLIR_SPARSETENSOR_FOREVERY_V(IMPL_LEXINSERT)
#undef IMPL_LEXINSERT

#define IMPL_EXPINSERT(VNAME, V)                                               \
  void _mlir_ciface_expInsert##VNAME(                                          \
      void *tensor, StridedMemRefType<index_type, 1> *cref,                    \
      StridedMemRefType<V, 1> *vref, StridedMemRefType<bool, 1> *fref,         \
      StridedMemRefType<index_type, 1> *aref, index_type count) {              \
    assert(tensor);                                                            \
    ASSERT_NO_STRIDE(cref);                                                    \
    ASSERT_NO_STRIDE(vref);                                                    \
    ASSERT_NO_STRIDE(fref);                                                    \
    ASSERT_NO_STRIDE(aref);                                                    \
    ASSERT_USIZE_EQ(vref, MEMREF_GET_USIZE(fref));                             \
    index_type *cursor = MEMREF_GET_PAYLOAD(cref);                             \
    V *values = MEMREF_GET_PAYLOAD(vref);                                      \
    bool *filled = MEMREF_GET_PAYLOAD(fref);                                   \
    index_type *added = MEMREF_GET_PAYLOAD(aref);                              \
    static_cast<SparseTensorStorageBase *>(tensor)->expInsert(                 \
        cursor, values, filled, added, count);                                 \
  }
MLIR_SPARSETENSOR_FOREVERY_V(IMPL_EXPINSERT)
#undef IMPL_EXPINSERT

void *_mlir_ciface_createCheckedSparseTensorReader(
    char *filename, StridedMemRefType<index_type, 1> *dimShapeRef,
    PrimaryType valTp) {
  ASSERT_NO_STRIDE(dimShapeRef);
  const uint64_t dimRank = MEMREF_GET_USIZE(dimShapeRef);
  const index_type *dimShape = MEMREF_GET_PAYLOAD(dimShapeRef);
  auto *reader = SparseTensorReader::create(filename, dimRank, dimShape, valTp);
  return static_cast<void *>(reader);
}

// FIXME: update `SparseTensorCodegenPass` to use
// `_mlir_ciface_getSparseTensorReaderDimSizes` instead.
void _mlir_ciface_copySparseTensorReaderDimSizes(
    void *p, StridedMemRefType<index_type, 1> *dref) {
  assert(p);
  SparseTensorReader &reader = *static_cast<SparseTensorReader *>(p);
  ASSERT_NO_STRIDE(dref);
  const uint64_t dimRank = MEMREF_GET_USIZE(dref);
  ASSERT_USIZE_EQ(dref, reader.getRank());
  index_type *dimSizes = MEMREF_GET_PAYLOAD(dref);
  const index_type *fileSizes = reader.getDimSizes();
  for (uint64_t d = 0; d < dimRank; ++d)
    dimSizes[d] = fileSizes[d];
}

void _mlir_ciface_getSparseTensorReaderDimSizes(
    StridedMemRefType<index_type, 1> *out, void *p) {
  assert(out && p);
  SparseTensorReader &reader = *static_cast<SparseTensorReader *>(p);
  auto *dimSizes = const_cast<uint64_t *>(reader.getDimSizes());
  aliasIntoMemref(reader.getRank(), dimSizes, *out);
}

#define IMPL_GETNEXT(VNAME, V)                                                 \
  void _mlir_ciface_getSparseTensorReaderNext##VNAME(                          \
      void *p, StridedMemRefType<index_type, 1> *iref,                         \
      StridedMemRefType<V, 0> *vref) {                                         \
    assert(p &&vref);                                                          \
    auto &reader = *static_cast<SparseTensorReader *>(p);                      \
    ASSERT_NO_STRIDE(iref);                                                    \
    const uint64_t rank = MEMREF_GET_USIZE(iref);                              \
    index_type *indices = MEMREF_GET_PAYLOAD(iref);                            \
    V *value = MEMREF_GET_PAYLOAD(vref);                                       \
    *value = reader.readCOOElement<V>(rank, indices);                          \
  }
MLIR_SPARSETENSOR_FOREVERY_V(IMPL_GETNEXT)
#undef IMPL_GETNEXT

void *_mlir_ciface_newSparseTensorFromReader(
    void *p, StridedMemRefType<index_type, 1> *lvlSizesRef,
    StridedMemRefType<DimLevelType, 1> *lvlTypesRef,
    StridedMemRefType<index_type, 1> *lvl2dimRef,
    StridedMemRefType<index_type, 1> *dim2lvlRef, OverheadType ptrTp,
    OverheadType indTp, PrimaryType valTp) {
  assert(p);
  SparseTensorReader &reader = *static_cast<SparseTensorReader *>(p);
  ASSERT_NO_STRIDE(lvlSizesRef);
  ASSERT_NO_STRIDE(lvlTypesRef);
  ASSERT_NO_STRIDE(lvl2dimRef);
  ASSERT_NO_STRIDE(dim2lvlRef);
  const uint64_t dimRank = reader.getRank();
  const uint64_t lvlRank = MEMREF_GET_USIZE(lvlSizesRef);
  ASSERT_USIZE_EQ(lvlTypesRef, lvlRank);
  ASSERT_USIZE_EQ(lvl2dimRef, lvlRank);
  ASSERT_USIZE_EQ(dim2lvlRef, dimRank);
  (void)dimRank;
  const index_type *lvlSizes = MEMREF_GET_PAYLOAD(lvlSizesRef);
  const DimLevelType *lvlTypes = MEMREF_GET_PAYLOAD(lvlTypesRef);
  const index_type *lvl2dim = MEMREF_GET_PAYLOAD(lvl2dimRef);
  const index_type *dim2lvl = MEMREF_GET_PAYLOAD(dim2lvlRef);
  //
  // FIXME(wrengr): Really need to define a separate x-macro for handling
  // all this. (Or ideally some better, entirely-different approach)
#define CASE(p, i, v, P, I, V)                                                 \
  if (ptrTp == OverheadType::p && indTp == OverheadType::i &&                  \
      valTp == PrimaryType::v)                                                 \
    return static_cast<void *>(reader.readSparseTensor<P, I, V>(               \
        lvlRank, lvlSizes, lvlTypes, lvl2dim, dim2lvl));
#define CASE_SECSAME(p, v, P, V) CASE(p, p, v, P, P, V)
  // Rewrite kIndex to kU64, to avoid introducing a bunch of new cases.
  // This is safe because of the static_assert above.
  if (ptrTp == OverheadType::kIndex)
    ptrTp = OverheadType::kU64;
  if (indTp == OverheadType::kIndex)
    indTp = OverheadType::kU64;
  // Double matrices with all combinations of overhead storage.
  CASE(kU64, kU64, kF64, uint64_t, uint64_t, double);
  CASE(kU64, kU32, kF64, uint64_t, uint32_t, double);
  CASE(kU64, kU16, kF64, uint64_t, uint16_t, double);
  CASE(kU64, kU8, kF64, uint64_t, uint8_t, double);
  CASE(kU32, kU64, kF64, uint32_t, uint64_t, double);
  CASE(kU32, kU32, kF64, uint32_t, uint32_t, double);
  CASE(kU32, kU16, kF64, uint32_t, uint16_t, double);
  CASE(kU32, kU8, kF64, uint32_t, uint8_t, double);
  CASE(kU16, kU64, kF64, uint16_t, uint64_t, double);
  CASE(kU16, kU32, kF64, uint16_t, uint32_t, double);
  CASE(kU16, kU16, kF64, uint16_t, uint16_t, double);
  CASE(kU16, kU8, kF64, uint16_t, uint8_t, double);
  CASE(kU8, kU64, kF64, uint8_t, uint64_t, double);
  CASE(kU8, kU32, kF64, uint8_t, uint32_t, double);
  CASE(kU8, kU16, kF64, uint8_t, uint16_t, double);
  CASE(kU8, kU8, kF64, uint8_t, uint8_t, double);
  // Float matrices with all combinations of overhead storage.
  CASE(kU64, kU64, kF32, uint64_t, uint64_t, float);
  CASE(kU64, kU32, kF32, uint64_t, uint32_t, float);
  CASE(kU64, kU16, kF32, uint64_t, uint16_t, float);
  CASE(kU64, kU8, kF32, uint64_t, uint8_t, float);
  CASE(kU32, kU64, kF32, uint32_t, uint64_t, float);
  CASE(kU32, kU32, kF32, uint32_t, uint32_t, float);
  CASE(kU32, kU16, kF32, uint32_t, uint16_t, float);
  CASE(kU32, kU8, kF32, uint32_t, uint8_t, float);
  CASE(kU16, kU64, kF32, uint16_t, uint64_t, float);
  CASE(kU16, kU32, kF32, uint16_t, uint32_t, float);
  CASE(kU16, kU16, kF32, uint16_t, uint16_t, float);
  CASE(kU16, kU8, kF32, uint16_t, uint8_t, float);
  CASE(kU8, kU64, kF32, uint8_t, uint64_t, float);
  CASE(kU8, kU32, kF32, uint8_t, uint32_t, float);
  CASE(kU8, kU16, kF32, uint8_t, uint16_t, float);
  CASE(kU8, kU8, kF32, uint8_t, uint8_t, float);
  // Two-byte floats with both overheads of the same type.
  CASE_SECSAME(kU64, kF16, uint64_t, f16);
  CASE_SECSAME(kU64, kBF16, uint64_t, bf16);
  CASE_SECSAME(kU32, kF16, uint32_t, f16);
  CASE_SECSAME(kU32, kBF16, uint32_t, bf16);
  CASE_SECSAME(kU16, kF16, uint16_t, f16);
  CASE_SECSAME(kU16, kBF16, uint16_t, bf16);
  CASE_SECSAME(kU8, kF16, uint8_t, f16);
  CASE_SECSAME(kU8, kBF16, uint8_t, bf16);
  // Integral matrices with both overheads of the same type.
  CASE_SECSAME(kU64, kI64, uint64_t, int64_t);
  CASE_SECSAME(kU64, kI32, uint64_t, int32_t);
  CASE_SECSAME(kU64, kI16, uint64_t, int16_t);
  CASE_SECSAME(kU64, kI8, uint64_t, int8_t);
  CASE_SECSAME(kU32, kI64, uint32_t, int64_t);
  CASE_SECSAME(kU32, kI32, uint32_t, int32_t);
  CASE_SECSAME(kU32, kI16, uint32_t, int16_t);
  CASE_SECSAME(kU32, kI8, uint32_t, int8_t);
  CASE_SECSAME(kU16, kI64, uint16_t, int64_t);
  CASE_SECSAME(kU16, kI32, uint16_t, int32_t);
  CASE_SECSAME(kU16, kI16, uint16_t, int16_t);
  CASE_SECSAME(kU16, kI8, uint16_t, int8_t);
  CASE_SECSAME(kU8, kI64, uint8_t, int64_t);
  CASE_SECSAME(kU8, kI32, uint8_t, int32_t);
  CASE_SECSAME(kU8, kI16, uint8_t, int16_t);
  CASE_SECSAME(kU8, kI8, uint8_t, int8_t);
  // Complex matrices with wide overhead.
  CASE_SECSAME(kU64, kC64, uint64_t, complex64);
  CASE_SECSAME(kU64, kC32, uint64_t, complex32);

  // Unsupported case (add above if needed).
  // TODO: better pretty-printing of enum values!
  MLIR_SPARSETENSOR_FATAL(
      "unsupported combination of types: <P=%d, I=%d, V=%d>\n",
      static_cast<int>(ptrTp), static_cast<int>(indTp),
      static_cast<int>(valTp));
#undef CASE_SECSAME
#undef CASE
}

void _mlir_ciface_outSparseTensorWriterMetaData(
    void *p, index_type rank, index_type nnz,
    StridedMemRefType<index_type, 1> *dref) {
  assert(p);
  ASSERT_NO_STRIDE(dref);
  assert(rank != 0);
  index_type *dimSizes = MEMREF_GET_PAYLOAD(dref);
  SparseTensorWriter &file = *static_cast<SparseTensorWriter *>(p);
  file << rank << " " << nnz << std::endl;
  for (index_type r = 0; r < rank - 1; ++r)
    file << dimSizes[r] << " ";
  file << dimSizes[rank - 1] << std::endl;
}

#define IMPL_OUTNEXT(VNAME, V)                                                 \
  void _mlir_ciface_outSparseTensorWriterNext##VNAME(                          \
      void *p, index_type rank, StridedMemRefType<index_type, 1> *iref,        \
      StridedMemRefType<V, 0> *vref) {                                         \
    assert(p &&vref);                                                          \
    ASSERT_NO_STRIDE(iref);                                                    \
    index_type *indices = MEMREF_GET_PAYLOAD(iref);                            \
    SparseTensorWriter &file = *static_cast<SparseTensorWriter *>(p);          \
    for (index_type r = 0; r < rank; ++r)                                      \
      file << (indices[r] + 1) << " ";                                         \
    V *value = MEMREF_GET_PAYLOAD(vref);                                       \
    file << *value << std::endl;                                               \
  }
MLIR_SPARSETENSOR_FOREVERY_V(IMPL_OUTNEXT)
#undef IMPL_OUTNEXT

//===----------------------------------------------------------------------===//
//
// Public functions which accept only C-style data structures to interact
// with sparse tensors (which are only visible as opaque pointers externally).
//
//===----------------------------------------------------------------------===//

index_type sparseLvlSize(void *tensor, index_type x) {
  return static_cast<SparseTensorStorageBase *>(tensor)->getLvlSize(x);
}

void endInsert(void *tensor) {
  return static_cast<SparseTensorStorageBase *>(tensor)->endInsert();
}

#define IMPL_OUTSPARSETENSOR(VNAME, V)                                         \
  void outSparseTensor##VNAME(void *coo, void *dest, bool sort) {              \
    assert(coo && "Got nullptr for COO object");                               \
    auto &coo_ = *static_cast<SparseTensorCOO<V> *>(coo);                      \
    if (sort)                                                                  \
      coo_.sort();                                                             \
    return writeExtFROSTT(coo_, static_cast<char *>(dest));                    \
  }
MLIR_SPARSETENSOR_FOREVERY_V(IMPL_OUTSPARSETENSOR)
#undef IMPL_OUTSPARSETENSOR

void delSparseTensor(void *tensor) {
  delete static_cast<SparseTensorStorageBase *>(tensor);
}

#define IMPL_DELCOO(VNAME, V)                                                  \
  void delSparseTensorCOO##VNAME(void *coo) {                                  \
    delete static_cast<SparseTensorCOO<V> *>(coo);                             \
  }
MLIR_SPARSETENSOR_FOREVERY_V(IMPL_DELCOO)
#undef IMPL_DELCOO

#define IMPL_DELITER(VNAME, V)                                                 \
  void delSparseTensorIterator##VNAME(void *iter) {                            \
    delete static_cast<SparseTensorIterator<V> *>(iter);                       \
  }
MLIR_SPARSETENSOR_FOREVERY_V(IMPL_DELITER)
#undef IMPL_DELITER

char *getTensorFilename(index_type id) {
  char var[80];
  sprintf(var, "TENSOR%" PRIu64, id);
  char *env = getenv(var);
  if (!env)
    MLIR_SPARSETENSOR_FATAL("Environment variable %s is not set\n", var);
  return env;
}

void readSparseTensorShape(char *filename, std::vector<uint64_t> *out) {
  assert(out && "Received nullptr for out-parameter");
  SparseTensorReader reader(filename);
  reader.openFile();
  reader.readHeader();
  reader.closeFile();
  const uint64_t dimRank = reader.getRank();
  const uint64_t *dimSizes = reader.getDimSizes();
  out->reserve(dimRank);
  out->assign(dimSizes, dimSizes + dimRank);
}

// We can't use `static_cast` here because `DimLevelType` is an enum-class.
#define IMPL_CONVERTTOMLIRSPARSETENSOR(VNAME, V)                               \
  void *convertToMLIRSparseTensor##VNAME(                                      \
      uint64_t rank, uint64_t nse, uint64_t *shape, V *values,                 \
      uint64_t *indices, uint64_t *perm, uint8_t *sparse) {                    \
    return toMLIRSparseTensor<V>(rank, nse, shape, values, indices, perm,      \
                                 reinterpret_cast<DimLevelType *>(sparse));    \
  }
MLIR_SPARSETENSOR_FOREVERY_V(IMPL_CONVERTTOMLIRSPARSETENSOR)
#undef IMPL_CONVERTTOMLIRSPARSETENSOR

#define IMPL_CONVERTFROMMLIRSPARSETENSOR(VNAME, V)                             \
  void convertFromMLIRSparseTensor##VNAME(void *tensor, uint64_t *pRank,       \
                                          uint64_t *pNse, uint64_t **pShape,   \
                                          V **pValues, uint64_t **pIndices) {  \
    fromMLIRSparseTensor<V>(                                                   \
        static_cast<SparseTensorStorage<uint64_t, uint64_t, V> *>(tensor),     \
        pRank, pNse, pShape, pValues, pIndices);                               \
  }
MLIR_SPARSETENSOR_FOREVERY_V(IMPL_CONVERTFROMMLIRSPARSETENSOR)
#undef IMPL_CONVERTFROMMLIRSPARSETENSOR

// FIXME: update `SparseTensorCodegenPass` to use
// `_mlir_ciface_createCheckedSparseTensorReader` instead.
void *createSparseTensorReader(char *filename) {
  SparseTensorReader *reader = new SparseTensorReader(filename);
  reader->openFile();
  reader->readHeader();
  return static_cast<void *>(reader);
}

index_type getSparseTensorReaderRank(void *p) {
  return static_cast<SparseTensorReader *>(p)->getRank();
}

bool getSparseTensorReaderIsSymmetric(void *p) {
  return static_cast<SparseTensorReader *>(p)->isSymmetric();
}

index_type getSparseTensorReaderNNZ(void *p) {
  return static_cast<SparseTensorReader *>(p)->getNNZ();
}

index_type getSparseTensorReaderDimSize(void *p, index_type d) {
  return static_cast<SparseTensorReader *>(p)->getDimSize(d);
}

void delSparseTensorReader(void *p) {
  delete static_cast<SparseTensorReader *>(p);
}

void *createSparseTensorWriter(char *filename) {
  SparseTensorWriter *file =
      (filename[0] == 0) ? &std::cout : new std::ofstream(filename);
  *file << "# extended FROSTT format\n";
  return static_cast<void *>(file);
}

void delSparseTensorWriter(void *p) {
  SparseTensorWriter *file = static_cast<SparseTensorWriter *>(p);
  file->flush();
  assert(file->good());
  if (file != &std::cout)
    delete file;
}

} // extern "C"

#undef MEMREF_GET_PAYLOAD
#undef ASSERT_USIZE_EQ
#undef MEMREF_GET_USIZE
#undef ASSERT_NO_STRIDE

#endif // MLIR_CRUNNERUTILS_DEFINE_FUNCTIONS
