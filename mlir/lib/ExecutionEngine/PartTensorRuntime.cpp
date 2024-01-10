//===- PartTensorRuntime.cpp - PartTensor runtime support lib ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a light-weight runtime support library for
// manipulating paritioned sparse tensors from MLIR.  More specifically, it
// provides C-API wrappers so that MLIR-generated code can call into the C++
// runtime support library.  The functionality provided in this library is meant
// to simplify benchmarking, testing, and debugging of MLIR code operating
// on sparse tensors.  However, the provided functionality is **not**
// part of core MLIR itself.
//
// The following memory-resident partitioned sparse storage schemes are
// supported:
//
// (a) A coordinate scheme for temporarily storing and lexicographically
//     sorting a sparse tensor by index (SparseTensorCOO).
//
//  // TODO: support other things supported by SparseTensor.
//
// The following external formats are supported:
//
// (1) Matrix Market Exchange (MME): *.mtx
//     https://math.nist.gov/MatrixMarket/formats.html
//
// Two public APIs are supported:
//
// (I) Methods operating on MLIR buffers (memrefs) to interact with partitioned
//     sparse tensors. These methods should be used exclusively by MLIR
//     compiler-generated code.
//
// (II) Methods that accept C-style data structures to interact with partitioned
//      sparse tensors. These methods can be used by any external runtime that
//      wants to interact with MLIR compiler-generated code.
//
// In both cases (I) and (II), the SparseTensorStorage format is externally
// only visible as an opaque pointer.
//
//===----------------------------------------------------------------------===//

#include "mlir/ExecutionEngine/PartTensorRuntime.h"
#include "mlir/Dialect/SparseTensor/IR/Enums.h"
#include "mlir/ExecutionEngine/PartTensor/Storage.h"
#include "mlir/ExecutionEngine/SparseTensorRuntime.h"
#include "llvm/ADT/ArrayRef.h"
#include <cassert>
#include <cstdint>

#ifdef MLIR_CRUNNERUTILS_DEFINE_FUNCTIONS

#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "mlir/ExecutionEngine/SparseTensor/ArithmeticUtils.h"
#include "mlir/ExecutionEngine/SparseTensor/COO.h"
#include "mlir/ExecutionEngine/SparseTensor/ErrorHandling.h"
#include "mlir/ExecutionEngine/SparseTensor/File.h"
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

} // anonymous namespace

extern "C" {

//===----------------------------------------------------------------------===//
//
// Public functions which operate on MLIR buffers (memrefs) to interact
// with sparse tensors (which are only visible as opaque pointers externally).
//
//===----------------------------------------------------------------------===//

#define CASE(v, V)                                                             \
  if (valTp == (v)) {                                                          \
    switch (action) {                                                          \
    case Action::kFromCOO: {                                                   \
      assert(ptr && "Received nullptr for PartTensorCOO object");              \
      auto coo = static_cast<SparseTensorCOO<V> *>(ptr);                       \
      return (void *)PartTensorStorage<uint64_t, uint64_t, V>::newFromCOO(     \
          partRank, partSizes, dimRank, dimSizes, lvlTypes, coo);              \
    }                                                                          \
    default:                                                                   \
      MLIR_SPARSETENSOR_FATAL("unknown action: %d\n",                          \
                              static_cast<uint32_t>(action));                  \
    }                                                                          \
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
void *
_mlir_ciface_newPartTensor(StridedMemRefType<index_type, 1> *partSizesRef,
                           StridedMemRefType<index_type, 1> *dimSizesRef,
                           StridedMemRefType<DimLevelType, 1> *lvlTypesRef,
                           PrimaryType valTp, Action action, void *ptr) {
  ASSERT_NO_STRIDE(dimSizesRef);
  const uint64_t partRank = MEMREF_GET_USIZE(partSizesRef);
  const uint64_t dimRank = MEMREF_GET_USIZE(dimSizesRef);
  const index_type *partSizes = MEMREF_GET_PAYLOAD(partSizesRef);
  const index_type *dimSizes = MEMREF_GET_PAYLOAD(dimSizesRef);
  const DimLevelType *lvlTypes = MEMREF_GET_PAYLOAD(lvlTypesRef);
  assert((valTp == PrimaryType::kF64 || valTp == PrimaryType::kF32) &&
         "Only float and double is supported for now");
  assert(action == mlir::sparse_tensor::Action::kFromCOO &&
         "Only kFromCOO is supported for now");

  // Double matrices with all combinations of overhead storage.
  CASE(PrimaryType::kF64, double);
  // Float matrices with all combinations of overhead storage.
  CASE(PrimaryType::kF32, float);

  // Unsupported case (add above if needed).
  // TODO: better pretty-printing of enum values!
  MLIR_SPARSETENSOR_FATAL("unsupported combination of types: <V=%d>\n",
                          static_cast<int>(valTp));
}
#undef CASE
#undef CASE_SECSAME

void _mlir_ciface_getPartitions( // NOLINT
    StridedMemRefType<index_type, 1> *partsMemRef, void *tensor) {
  std::vector<index_type> *parts;
  static_cast<PartTensorStorageBase *>(tensor)->getPartitions(&parts);
  aliasIntoMemref(parts->size(), parts->data(), *partsMemRef);
}

index_type _mlir_ciface_getNumPartitions(void *tensor) {
  return static_cast<PartTensorStorageBase *>(tensor)->getNumPartitions();
}

void *_mlir_ciface_getSlice(void *tensor,
                            StridedMemRefType<index_type, 1> *partSpec) {
  return static_cast<PartTensorStorageBase *>(tensor)->getSlice(
      llvm::ArrayRef<index_type>(partSpec->data + partSpec->offset,
                                 partSpec->sizes[0]));
}
void _mlir_ciface_setSlice(void *tensor,
                           StridedMemRefType<index_type, 1> *partSpec,
                           void *spTensor) {
  static_cast<PartTensorStorageBase *>(tensor)->setSlice(
      llvm::ArrayRef<index_type>(partSpec->data + partSpec->offset,
                                 partSpec->sizes[0]),
      static_cast<SparseTensorStorageBase *>(spTensor));
}
extern void *snl_utah_spadd_dense_f32(void *tensor, void *spTensor);

void _mlir_ciface_updateSlice(void *partTensor,
                              StridedMemRefType<index_type, 1> *partSpec,
                              void *spTensor) {
  // For now it only works on dense.
  auto *oldVal = static_cast<SparseTensorStorageBase *>(
      _mlir_ciface_getSlice(partTensor, partSpec));
  auto *newVal = static_cast<SparseTensorStorageBase *>(spTensor);
  std::vector<float> *valuesVector;
  std::vector<float> *newValuesVector;
  oldVal->getValues(&valuesVector);
  newVal->getValues(&newValuesVector);
  for (auto elem : llvm::zip(*valuesVector, *newValuesVector)) {
    std::get<0>(elem) += std::get<1>(elem);
  }
}
} // extern "C"

#undef MEMREF_GET_PAYLOAD
#undef ASSERT_USIZE_EQ
#undef MEMREF_GET_USIZE
#undef ASSERT_NO_STRIDE

#endif // MLIR_CRUNNERUTILS_DEFINE_FUNCTIONS
