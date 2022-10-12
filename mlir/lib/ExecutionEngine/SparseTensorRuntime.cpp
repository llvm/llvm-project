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

#include "mlir/ExecutionEngine/SparseTensor/COO.h"
#include "mlir/ExecutionEngine/SparseTensor/ErrorHandling.h"
#include "mlir/ExecutionEngine/SparseTensor/File.h"
#include "mlir/ExecutionEngine/SparseTensor/Storage.h"

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

/// Initializes sparse tensor from an external COO-flavored format.
/// Used by `IMPL_CONVERTTOMLIRSPARSETENSOR`.
// TODO: generalize beyond 64-bit indices.
template <typename V>
static SparseTensorStorage<uint64_t, uint64_t, V> *
toMLIRSparseTensor(uint64_t rank, uint64_t nse, const uint64_t *shape,
                   const V *values, const uint64_t *indices,
                   const uint64_t *perm, const DimLevelType *sparsity) {
#ifndef NDEBUG
  // Verify that perm is a permutation of 0..(rank-1).
  std::vector<bool> seen(rank, false);
  for (uint64_t i = 0; i < rank; ++i) {
    const uint64_t j = perm[i];
    if (j >= rank || seen[j])
      MLIR_SPARSETENSOR_FATAL("Not a permutation of 0..%" PRIu64 "\n", rank);
    seen[j] = true;
  }

  // Verify that the sparsity values are supported.
  // TODO: update this check to match what we actually support.
  for (uint64_t i = 0; i < rank; ++i)
    if (sparsity[i] != DimLevelType::kDense &&
        sparsity[i] != DimLevelType::kCompressed)
      MLIR_SPARSETENSOR_FATAL("unsupported dimension level type: %d\n",
                              static_cast<uint8_t>(sparsity[i]));
#endif

  // Convert external format to internal COO.
  auto *coo = SparseTensorCOO<V>::newSparseTensorCOO(rank, shape, perm, nse);
  std::vector<uint64_t> idx(rank);
  for (uint64_t i = 0, base = 0; i < nse; i++) {
    for (uint64_t r = 0; r < rank; r++)
      idx[perm[r]] = indices[base + r];
    coo->add(idx, values[i]);
    base += rank;
  }
  // Return sparse tensor storage format as opaque pointer.
  auto *tensor = SparseTensorStorage<uint64_t, uint64_t, V>::newSparseTensor(
      rank, shape, perm, sparsity, coo);
  delete coo;
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
  uint64_t rank = tensor->getRank();
  std::vector<uint64_t> perm(rank);
  std::iota(perm.begin(), perm.end(), 0);
  SparseTensorCOO<V> *coo = tensor->toCOO(perm.data());

  const std::vector<Element<V>> &elements = coo->getElements();
  uint64_t nse = elements.size();

  uint64_t *shape = new uint64_t[rank];
  for (uint64_t i = 0; i < rank; i++)
    shape[i] = coo->getDimSizes()[i];

  V *values = new V[nse];
  uint64_t *indices = new uint64_t[rank * nse];

  for (uint64_t i = 0, base = 0; i < nse; i++) {
    values[i] = elements[i].value;
    for (uint64_t j = 0; j < rank; j++)
      indices[base + j] = elements[i].indices[j];
    base += rank;
  }

  delete coo;
  *pRank = rank;
  *pNse = nse;
  *pShape = shape;
  *pValues = values;
  *pIndices = indices;
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
    SparseTensorCOO<V> *coo = nullptr;                                         \
    if (action <= Action::kFromCOO) {                                          \
      if (action == Action::kFromFile) {                                       \
        char *filename = static_cast<char *>(ptr);                             \
        coo = openSparseTensorCOO<V>(filename, rank, shape, perm, v);          \
      } else if (action == Action::kFromCOO) {                                 \
        coo = static_cast<SparseTensorCOO<V> *>(ptr);                          \
      } else {                                                                 \
        assert(action == Action::kEmpty);                                      \
      }                                                                        \
      auto *tensor = SparseTensorStorage<P, I, V>::newSparseTensor(            \
          rank, shape, perm, sparsity, coo);                                   \
      if (action == Action::kFromFile)                                         \
        delete coo;                                                            \
      return tensor;                                                           \
    }                                                                          \
    if (action == Action::kSparseToSparse) {                                   \
      auto *tensor = static_cast<SparseTensorStorageBase *>(ptr);              \
      return SparseTensorStorage<P, I, V>::newSparseTensor(rank, shape, perm,  \
                                                           sparsity, tensor);  \
    }                                                                          \
    if (action == Action::kEmptyCOO)                                           \
      return SparseTensorCOO<V>::newSparseTensorCOO(rank, shape, perm);        \
    coo = static_cast<SparseTensorStorage<P, I, V> *>(ptr)->toCOO(perm);       \
    if (action == Action::kToIterator) {                                       \
      return new SparseTensorIterator<V>(coo);                                 \
    } else {                                                                   \
      assert(action == Action::kToCOO);                                        \
    }                                                                          \
    return coo;                                                                \
  }

#define CASE_SECSAME(p, v, P, V) CASE(p, p, v, P, P, V)

// Assume index_type is in fact uint64_t, so that _mlir_ciface_newSparseTensor
// can safely rewrite kIndex to kU64.  We make this assertion to guarantee
// that this file cannot get out of sync with its header.
static_assert(std::is_same<index_type, uint64_t>::value,
              "Expected index_type == uint64_t");

void *
_mlir_ciface_newSparseTensor(StridedMemRefType<DimLevelType, 1> *aref, // NOLINT
                             StridedMemRefType<index_type, 1> *sref,
                             StridedMemRefType<index_type, 1> *pref,
                             OverheadType ptrTp, OverheadType indTp,
                             PrimaryType valTp, Action action, void *ptr) {
  assert(aref && sref && pref);
  assert(aref->strides[0] == 1 && sref->strides[0] == 1 &&
         pref->strides[0] == 1);
  assert(aref->sizes[0] == sref->sizes[0] && sref->sizes[0] == pref->sizes[0]);
  const DimLevelType *sparsity = aref->data + aref->offset;
  const index_type *shape = sref->data + sref->offset;
  const index_type *perm = pref->data + pref->offset;
  uint64_t rank = aref->sizes[0];

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
    ref->basePtr = ref->data = v->data();                                      \
    ref->offset = 0;                                                           \
    ref->sizes[0] = v->size();                                                 \
    ref->strides[0] = 1;                                                       \
  }
MLIR_SPARSETENSOR_FOREVERY_V(IMPL_SPARSEVALUES)
#undef IMPL_SPARSEVALUES

#define IMPL_GETOVERHEAD(NAME, TYPE, LIB)                                      \
  void _mlir_ciface_##NAME(StridedMemRefType<TYPE, 1> *ref, void *tensor,      \
                           index_type d) {                                     \
    assert(ref &&tensor);                                                      \
    std::vector<TYPE> *v;                                                      \
    static_cast<SparseTensorStorageBase *>(tensor)->LIB(&v, d);                \
    ref->basePtr = ref->data = v->data();                                      \
    ref->offset = 0;                                                           \
    ref->sizes[0] = v->size();                                                 \
    ref->strides[0] = 1;                                                       \
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

#define IMPL_ADDELT(VNAME, V)                                                  \
  void *_mlir_ciface_addElt##VNAME(void *coo, StridedMemRefType<V, 0> *vref,   \
                                   StridedMemRefType<index_type, 1> *iref,     \
                                   StridedMemRefType<index_type, 1> *pref) {   \
    assert(coo &&vref &&iref &&pref);                                          \
    assert(iref->strides[0] == 1 && pref->strides[0] == 1);                    \
    assert(iref->sizes[0] == pref->sizes[0]);                                  \
    const index_type *indx = iref->data + iref->offset;                        \
    const index_type *perm = pref->data + pref->offset;                        \
    uint64_t isize = iref->sizes[0];                                           \
    std::vector<index_type> indices(isize);                                    \
    for (uint64_t r = 0; r < isize; r++)                                       \
      indices[perm[r]] = indx[r];                                              \
    V *value = vref->data + vref->offset;                                      \
    static_cast<SparseTensorCOO<V> *>(coo)->add(indices, *value);              \
    return coo;                                                                \
  }
MLIR_SPARSETENSOR_FOREVERY_V(IMPL_ADDELT)
#undef IMPL_ADDELT

#define IMPL_GETNEXT(VNAME, V)                                                 \
  bool _mlir_ciface_getNext##VNAME(void *iter,                                 \
                                   StridedMemRefType<index_type, 1> *iref,     \
                                   StridedMemRefType<V, 0> *vref) {            \
    assert(iter &&iref &&vref);                                                \
    assert(iref->strides[0] == 1);                                             \
    index_type *indx = iref->data + iref->offset;                              \
    V *value = vref->data + vref->offset;                                      \
    const uint64_t isize = iref->sizes[0];                                     \
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
    assert(tensor &&cref &&vref);                                              \
    assert(cref->strides[0] == 1);                                             \
    index_type *cursor = cref->data + cref->offset;                            \
    assert(cursor);                                                            \
    V *value = vref->data + vref->offset;                                      \
    static_cast<SparseTensorStorageBase *>(tensor)->lexInsert(cursor, *value); \
  }
MLIR_SPARSETENSOR_FOREVERY_V(IMPL_LEXINSERT)
#undef IMPL_LEXINSERT

#define IMPL_EXPINSERT(VNAME, V)                                               \
  void _mlir_ciface_expInsert##VNAME(                                          \
      void *tensor, StridedMemRefType<index_type, 1> *cref,                    \
      StridedMemRefType<V, 1> *vref, StridedMemRefType<bool, 1> *fref,         \
      StridedMemRefType<index_type, 1> *aref, index_type count) {              \
    assert(tensor &&cref &&vref &&fref &&aref);                                \
    assert(cref->strides[0] == 1);                                             \
    assert(vref->strides[0] == 1);                                             \
    assert(fref->strides[0] == 1);                                             \
    assert(aref->strides[0] == 1);                                             \
    assert(vref->sizes[0] == fref->sizes[0]);                                  \
    index_type *cursor = cref->data + cref->offset;                            \
    V *values = vref->data + vref->offset;                                     \
    bool *filled = fref->data + fref->offset;                                  \
    index_type *added = aref->data + aref->offset;                             \
    static_cast<SparseTensorStorageBase *>(tensor)->expInsert(                 \
        cursor, values, filled, added, count);                                 \
  }
MLIR_SPARSETENSOR_FOREVERY_V(IMPL_EXPINSERT)
#undef IMPL_EXPINSERT

//===----------------------------------------------------------------------===//
//
// Public functions which accept only C-style data structures to interact
// with sparse tensors (which are only visible as opaque pointers externally).
//
//===----------------------------------------------------------------------===//

index_type sparseDimSize(void *tensor, index_type d) {
  return static_cast<SparseTensorStorageBase *>(tensor)->getDimSize(d);
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
  SparseTensorReader stfile(filename);
  stfile.openFile();
  stfile.readHeader();
  stfile.closeFile();
  const uint64_t rank = stfile.getRank();
  const uint64_t *dimSizes = stfile.getDimSizes();
  out->reserve(rank);
  out->assign(dimSizes, dimSizes + rank);
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

void *createSparseTensorReader(char *filename) {
  SparseTensorReader *stfile = new SparseTensorReader(filename);
  stfile->openFile();
  stfile->readHeader();
  return static_cast<void *>(stfile);
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

void _mlir_ciface_getSparseTensorReaderDimSizes(
    void *p, StridedMemRefType<index_type, 1> *dref) {
  assert(p && dref);
  assert(dref->strides[0] == 1);
  index_type *dimSizes = dref->data + dref->offset;
  SparseTensorReader &file = *static_cast<SparseTensorReader *>(p);
  const index_type *sizes = file.getDimSizes();
  index_type rank = file.getRank();
  for (uint64_t r = 0; r < rank; ++r)
    dimSizes[r] = sizes[r];
}

void delSparseTensorReader(void *p) {
  delete static_cast<SparseTensorReader *>(p);
}

#define IMPL_GETNEXT(VNAME, V)                                                 \
  V _mlir_ciface_getSparseTensorReaderNext##VNAME(                             \
      void *p, StridedMemRefType<index_type, 1> *iref) {                       \
    assert(p &&iref);                                                          \
    assert(iref->strides[0] == 1);                                             \
    index_type *indices = iref->data + iref->offset;                           \
    SparseTensorReader *stfile = static_cast<SparseTensorReader *>(p);         \
    index_type rank = stfile->getRank();                                       \
    char *linePtr = stfile->readLine();                                        \
    for (index_type r = 0; r < rank; ++r) {                                    \
      uint64_t idx = strtoul(linePtr, &linePtr, 10);                           \
      indices[r] = idx - 1;                                                    \
    }                                                                          \
    return detail::readCOOValue<V>(&linePtr, stfile->isPattern());             \
  }
MLIR_SPARSETENSOR_FOREVERY_V(IMPL_GETNEXT)
#undef IMPL_GETNEXT

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

void _mlir_ciface_outSparseTensorWriterMetaData(
    void *p, index_type rank, index_type nnz,
    StridedMemRefType<index_type, 1> *dref) {
  assert(p && dref);
  assert(dref->strides[0] == 1);
  assert(rank != 0);
  index_type *dimSizes = dref->data + dref->offset;
  SparseTensorWriter &file = *static_cast<SparseTensorWriter *>(p);
  file << rank << " " << nnz << std::endl;
  for (index_type r = 0; r < rank - 1; ++r)
    file << dimSizes[r] << " ";
  file << dimSizes[rank - 1] << std::endl;
}

#define IMPL_OUTNEXT(VNAME, V)                                                 \
  void _mlir_ciface_outSparseTensorWriterNext##VNAME(                          \
      void *p, index_type rank, StridedMemRefType<index_type, 1> *iref,        \
      V value) {                                                               \
    assert(p &&iref);                                                          \
    assert(iref->strides[0] == 1);                                             \
    index_type *indices = iref->data + iref->offset;                           \
    SparseTensorWriter &file = *static_cast<SparseTensorWriter *>(p);          \
    for (uint64_t r = 0; r < rank; ++r)                                        \
      file << (indices[r] + 1) << " ";                                         \
    file << value << std::endl;                                                \
  }
MLIR_SPARSETENSOR_FOREVERY_V(IMPL_OUTNEXT)
#undef IMPL_OUTNEXT

} // extern "C"

#endif // MLIR_CRUNNERUTILS_DEFINE_FUNCTIONS
