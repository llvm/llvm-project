//===- Matrix.h - Two-dimensional Container with View -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_MATRIX_H
#define LLVM_ADT_MATRIX_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Compiler.h"

namespace llvm {
template <typename T, size_t M, size_t NStorageInline> class JaggedArrayView;

/// Due to the SmallVector infrastructure using SmallVectorAlignmentAndOffset
/// that depends on the exact data layout, no derived classes can have extra
/// members.
template <typename T, size_t N>
struct MatrixStorageBase : public SmallVectorImpl<T>, SmallVectorStorage<T, N> {
  LLVM_ATTRIBUTE_ALWAYS_INLINE MatrixStorageBase() : SmallVectorImpl<T>(N) {}
  LLVM_ATTRIBUTE_ALWAYS_INLINE MatrixStorageBase(size_t Size)
      : SmallVectorImpl<T>(N) {
    resize(Size);
  }
  LLVM_ATTRIBUTE_ALWAYS_INLINE ~MatrixStorageBase() {
    destroy_range(this->begin(), this->end());
  }
  LLVM_ATTRIBUTE_ALWAYS_INLINE MatrixStorageBase(const MatrixStorageBase &RHS)
      : SmallVectorImpl<T>(N) {
    if (!RHS.empty())
      SmallVectorImpl<T>::operator=(RHS);
  }
  LLVM_ATTRIBUTE_ALWAYS_INLINE MatrixStorageBase(MatrixStorageBase &&RHS)
      : SmallVectorImpl<T>(N) {
    if (!RHS.empty())
      SmallVectorImpl<T>::operator=(::std::move(RHS));
  }
  using SmallVectorImpl<T>::size;
  using SmallVectorImpl<T>::resize;
  using SmallVectorImpl<T>::append;
  using SmallVectorImpl<T>::erase;
  using SmallVectorImpl<T>::destroy_range;
  using SmallVectorImpl<T>::isSafeToReferenceAfterResize;

  LLVM_ATTRIBUTE_ALWAYS_INLINE T *begin() const {
    return const_cast<T *>(SmallVectorImpl<T>::begin());
  }
  LLVM_ATTRIBUTE_ALWAYS_INLINE T *end() const {
    return const_cast<T *>(SmallVectorImpl<T>::end());
  }
};

/// A two-dimensional container storage, whose upper bound on the number of
/// columns should be known ahead of time. Not menat to be used directly: the
/// primary usage API is MatrixView.
template <typename T,
          size_t N = CalculateSmallVectorDefaultInlinedElements<T>::value>
class MatrixStorage {
public:
  MatrixStorage() = delete;
  MatrixStorage(size_t NRows, size_t NCols)
      : Base(NRows * NCols), NCols(NCols) {}
  MatrixStorage(size_t NCols) : Base(), NCols(NCols) {}

  LLVM_ATTRIBUTE_ALWAYS_INLINE size_t size() const { return Base.size(); }
  LLVM_ATTRIBUTE_ALWAYS_INLINE bool empty() const { return !size(); }
  LLVM_ATTRIBUTE_ALWAYS_INLINE size_t getNumRows() const {
    assert(size() % NCols == 0);
    return size() / NCols;
  }
  LLVM_ATTRIBUTE_ALWAYS_INLINE size_t getNumCols() const { return NCols; }
  LLVM_ATTRIBUTE_ALWAYS_INLINE void setNumCols(size_t NCols) {
    assert(empty() && "Column-resizing a non-empty MatrixStorage");
    this->NCols = NCols;
  }
  LLVM_ATTRIBUTE_ALWAYS_INLINE void resize(size_t NRows) {
    Base.resize(NCols * NRows);
  }
  LLVM_ATTRIBUTE_ALWAYS_INLINE void reserve(size_t NRows) {
    Base.reserve(NCols * NRows);
  }

protected:
  template <typename U, size_t M, size_t NStorageInline>
  friend class JaggedArrayView;

  LLVM_ATTRIBUTE_ALWAYS_INLINE T *begin() const { return Base.begin(); }
  LLVM_ATTRIBUTE_ALWAYS_INLINE T *rowFromIdx(size_t RowIdx,
                                             size_t Offset = 0) const {
    assert(Offset < NCols);
    return begin() + RowIdx * NCols + Offset;
  }
  LLVM_ATTRIBUTE_ALWAYS_INLINE std::pair<size_t, size_t>
  idxFromRow(T *Ptr) const {
    assert(Ptr >= begin());
    size_t Offset = (Ptr - begin()) % NCols;
    return {(Ptr - begin()) / NCols, Offset};
  }

  // If Arg.size() < NCols, the number of columns won't be changed, and the
  // difference is default-constructed.
  LLVM_ATTRIBUTE_ALWAYS_INLINE void addRow(const SmallVectorImpl<T> &Arg) {
    assert(Arg.size() <= NCols &&
           "MatrixStorage has insufficient number of columns");
    size_t Diff = NCols - Arg.size();
    Base.append(Arg.begin(), Arg.end());
    Base.append(Diff, T());
  }

  LLVM_ATTRIBUTE_ALWAYS_INLINE void eraseLastRow() {
    assert(getNumRows() > 0 && "Non-empty MatrixStorage expected");
    Base.pop_back_n(NCols);
  }

  LLVM_ATTRIBUTE_ALWAYS_INLINE bool willReallocateOnAddRow() const {
    return Base.capacity() < Base.size() + NCols;
  }

private:
  MatrixStorageBase<T, N> Base;
  size_t NCols;
};

/// MutableArrayRef with a copy-assign, and extra APIs.
template <typename T>
struct [[nodiscard]] MutableRowView : public MutableArrayRef<T> {
  using pointer = typename MutableArrayRef<T>::pointer;
  using iterator = typename MutableArrayRef<T>::iterator;
  using const_iterator = typename MutableArrayRef<T>::const_iterator;

  MutableRowView() = delete;
  MutableRowView(pointer Data, size_t Length)
      : MutableArrayRef<T>(Data, Length) {}
  MutableRowView(iterator Begin, iterator End)
      : MutableArrayRef<T>(Begin, End) {}
  MutableRowView(const_iterator Begin, const_iterator End)
      : MutableArrayRef<T>(Begin, End) {}
  MutableRowView(MutableArrayRef<T> Other)
      : MutableArrayRef<T>(Other.data(), Other.size()) {}
  MutableRowView(SmallVectorImpl<T> &Vec) : MutableArrayRef<T>(Vec) {}

  using MutableArrayRef<T>::size;
  using MutableArrayRef<T>::data;
  using MutableArrayRef<T>::begin;
  using MutableArrayRef<T>::end;

  LLVM_ATTRIBUTE_ALWAYS_INLINE T &back() const {
    return MutableArrayRef<T>::back();
  }
  LLVM_ATTRIBUTE_ALWAYS_INLINE T &front() const {
    return MutableArrayRef<T>::front();
  }
  LLVM_ATTRIBUTE_ALWAYS_INLINE MutableRowView<T>
  drop_back(size_t N = 1) const { // NOLINT
    return MutableArrayRef<T>::drop_back(N);
  }
  LLVM_ATTRIBUTE_ALWAYS_INLINE MutableRowView<T>
  drop_front(size_t N = 1) const { // NOLINT
    return MutableArrayRef<T>::drop_front(N);
  }
  // This slice is different from the MutableArrayRef slice, and specifies a
  // Begin and End index, instead of a Begin and Length.
  LLVM_ATTRIBUTE_ALWAYS_INLINE MutableRowView<T> slice(size_t Begin,
                                                       size_t End) {
    return MutableArrayRef<T>::slice(Begin, End - Begin);
  }
  LLVM_ATTRIBUTE_ALWAYS_INLINE void pop_back(size_t N = 1) { // NOLINT
    this->Length -= N;
  }
  LLVM_ATTRIBUTE_ALWAYS_INLINE void pop_front(size_t N = 1) { // NOLINT
    this->Data += N;
    this->Length -= N;
  }

  LLVM_ATTRIBUTE_ALWAYS_INLINE MutableRowView &
  operator=(const SmallVectorImpl<T> &Vec) {
    copy_assign(Vec.begin(), Vec.end());
    return *this;
  }
  LLVM_ATTRIBUTE_ALWAYS_INLINE MutableRowView &
  operator=(std::initializer_list<T> IL) {
    copy_assign(IL.begin(), IL.end());
    return *this;
  }

  LLVM_ATTRIBUTE_ALWAYS_INLINE void swap(MutableRowView<T> &Other) {
    std::swap(this->Data, Other.Data);
    std::swap(this->Length, Other.Length);
  }

  // For better cache behavior.
  LLVM_ATTRIBUTE_ALWAYS_INLINE void
  copy_assign(const MutableRowView<T> &Other) { // NOLINT
    copy_assign(Other.begin(), Other.end());
  }

  // For better cache behavior.
  LLVM_ATTRIBUTE_ALWAYS_INLINE void
  copy_swap(MutableRowView<T> &Other) { // NOLINT
    SmallVector<T> Buf{Other};
    Other.copy_assign(begin(), end());
    copy_assign(Buf.begin(), Buf.end());
  }

protected:
  LLVM_ATTRIBUTE_ALWAYS_INLINE void copy_assign(iterator Begin,
                                                iterator End) { // NOLINT
    std::uninitialized_copy(Begin, End, data());
    this->Length = End - Begin;
  }
  LLVM_ATTRIBUTE_ALWAYS_INLINE void copy_assign(const_iterator Begin,
                                                const_iterator End) { // NOLINT
    std::uninitialized_copy(Begin, End, data());
    this->Length = End - Begin;
  }
};

/// The primary usage API of MatrixStorage. Abstracts out indexing-arithmetic,
/// eliminating memory operations on the underlying data. Supports
/// variable-length columns.
template <typename T,
          size_t N = CalculateSmallVectorDefaultInlinedElements<T>::value,
          size_t NStorageInline =
              CalculateSmallVectorDefaultInlinedElements<T>::value>
class [[nodiscard]] JaggedArrayView {
public:
  using row_type = MutableRowView<T>;
  using container_type = SmallVector<row_type, N>;
  using iterator = typename container_type::iterator;
  using const_iterator = typename container_type::const_iterator;

  LLVM_ATTRIBUTE_ALWAYS_INLINE constexpr JaggedArrayView(
      MatrixStorage<T, NStorageInline> &Mat, size_t RowSpan, size_t ColSpan)
      : Mat(Mat) {
    RowView.reserve(RowSpan);
    for (size_t RowIdx = 0; RowIdx < RowSpan; ++RowIdx) {
      auto RangeBegin = Mat.begin() + RowIdx * ColSpan;
      RowView.emplace_back(RangeBegin, RangeBegin + ColSpan);
    }
  }

  // Constructor with a full View of the underlying MatrixStorage, if
  // MatrixStorage has a non-zero number of Columns. Otherwise, creates an empty
  // view.
  LLVM_ATTRIBUTE_ALWAYS_INLINE constexpr JaggedArrayView(
      MatrixStorage<T, NStorageInline> &Mat)
      : JaggedArrayView(Mat, Mat.getNumRows(), Mat.getNumCols()) {}

  // Obvious copy-construator is deleted, since the underlying storage could
  // have changed.
  constexpr JaggedArrayView(const JaggedArrayView &) = delete;

  // Copy-assignment operator should not be used when the underlying storage
  // changes.
  LLVM_ATTRIBUTE_ALWAYS_INLINE constexpr JaggedArrayView &
  operator=(const JaggedArrayView &Other) {
    assert(Mat.begin() == Other.Mat.begin() &&
           "Underlying storage has changed: use custom copy-constructor");
    RowView = Other.RowView;
    return *this;
  }

  // The actual copy-constructor: to be used when the underlying storage is
  // copy-constructed.
  JaggedArrayView(const JaggedArrayView &OldView,
                  MatrixStorage<T, NStorageInline> &NewMat)
      : Mat(NewMat) {
    assert(OldView.Mat.size() == Mat.size() &&
           "Custom copy-constructor called on non-copied storage");

    // The underlying storage will change. Construct a new RowView by performing
    // pointer-arithmetic on the underlying storage of OldView, using pointers
    // from OldVie.
    for (const auto &R : OldView.RowView) {
      auto [StorageIdx, StartOffset] = OldView.Mat.idxFromRow(R.data());
      RowView.emplace_back(Mat.rowFromIdx(StorageIdx, StartOffset), R.size());
    }
  }

  void addRow(const SmallVectorImpl<T> &Row) {
    // Optimization when we know that the underying storage won't be resized.
    if (LLVM_LIKELY(!Mat.willReallocateOnAddRow())) {
      Mat.addRow(Row);
      RowView.emplace_back(Mat.rowFromIdx(Mat.getNumRows() - 1), Row.size());
      return;
    }

    // The underlying storage may be resized, performing reallocations. The
    // pointers in RowView will no longer be valid, so save and restore the
    // data. Construct RestoreData by performing pointer-arithmetic on the
    // underlying storgge.
    SmallVector<std::tuple<size_t, size_t, size_t>> RestoreData;
    RestoreData.reserve(RowView.size());
    for (const auto &R : RowView) {
      auto [StorageIdx, StartOffset] = Mat.idxFromRow(R.data());
      RestoreData.emplace_back(StorageIdx, StartOffset, R.size());
    }

    Mat.addRow(Row);

    // Restore the RowView by performing pointer-arithmetic on the
    // possibly-reallocated storage, using information from RestoreData.
    RowView.clear();
    for (const auto &[StorageIdx, StartOffset, Len] : RestoreData)
      RowView.emplace_back(Mat.rowFromIdx(StorageIdx, StartOffset), Len);

    // Finally, add the new row to the VRowView.
    RowView.emplace_back(Mat.rowFromIdx(Mat.getNumRows() - 1), Row.size());
  }

  // To support addRow(View[Idx]).
  LLVM_ATTRIBUTE_ALWAYS_INLINE void addRow(const row_type &Row) {
    addRow(SmallVector<T>{Row});
  }

  LLVM_ATTRIBUTE_ALWAYS_INLINE void addRow(std::initializer_list<T> Row) {
    addRow(SmallVector<T>{Row});
  }

  LLVM_ATTRIBUTE_ALWAYS_INLINE constexpr row_type &operator[](size_t RowIdx) {
    assert(RowIdx < RowView.size() && "Indexing out of bounds");
    return RowView[RowIdx];
  }

  LLVM_ATTRIBUTE_ALWAYS_INLINE constexpr T *data() const {
    assert(!empty() && "Non-empty view expected");
    return RowView.front().data();
  }
  size_t size() const { return getRowSpan() * getMaxColSpan(); }
  LLVM_ATTRIBUTE_ALWAYS_INLINE bool empty() const { return RowView.empty(); }
  LLVM_ATTRIBUTE_ALWAYS_INLINE size_t getRowSpan() const {
    return RowView.size();
  }
  LLVM_ATTRIBUTE_ALWAYS_INLINE size_t getColSpan(size_t RowIdx) const {
    assert(RowIdx < RowView.size() && "Indexing out of bounds");
    return RowView[RowIdx].size();
  }
  constexpr size_t getMaxColSpan() const {
    return std::max_element(RowView.begin(), RowView.end(),
                            [](const row_type &RowA, const row_type &RowB) {
                              return RowA.size() < RowB.size();
                            })
        ->size();
  }

  LLVM_ATTRIBUTE_ALWAYS_INLINE iterator begin() { return RowView.begin(); }
  LLVM_ATTRIBUTE_ALWAYS_INLINE iterator end() { return RowView.end(); }
  LLVM_ATTRIBUTE_ALWAYS_INLINE const_iterator begin() const {
    return RowView.begin();
  }
  LLVM_ATTRIBUTE_ALWAYS_INLINE const_iterator end() const {
    return RowView.end();
  }

  constexpr JaggedArrayView<T, N, NStorageInline> rowSlice(size_t Begin,
                                                           size_t End) {
    assert(Begin < getRowSpan() && End <= getRowSpan() &&
           "Indexing out of bounds");
    assert(Begin < End && "Invalid slice");
    container_type NewRowView;
    for (size_t RowIdx = Begin; RowIdx < End; ++RowIdx)
      NewRowView.emplace_back(RowView[RowIdx]);
    return {Mat, std::move(NewRowView)};
  }

  constexpr JaggedArrayView<T, N, NStorageInline> colSlice(size_t Begin,
                                                           size_t End) {
    assert(Begin < End && "Invalid slice");
    size_t MinColSpan =
        std::min_element(RowView.begin(), RowView.end(),
                         [](const row_type &RowA, const row_type &RowB) {
                           return RowA.size() < RowB.size();
                         })
            ->size();
    assert(Begin < MinColSpan && End <= MinColSpan && "Indexing out of bounds");
    container_type NewRowView;
    for (row_type Row : RowView)
      NewRowView.emplace_back(Row.slice(Begin, End));
    return {Mat, std::move(NewRowView)};
  }

  LLVM_ATTRIBUTE_ALWAYS_INLINE row_type &lastRow() {
    assert(!empty() && "Non-empty view expected");
    return RowView.back();
  }
  LLVM_ATTRIBUTE_ALWAYS_INLINE const row_type &lastRow() const {
    assert(!empty() && "Non-empty view expected");
    return RowView.back();
  }
  LLVM_ATTRIBUTE_ALWAYS_INLINE void dropLastRow() {
    assert(!empty() && "Non-empty view expected");
    RowView.pop_back();
  }

  // For better cache behavior. To be used with copy_assign or copy_swap.
  LLVM_ATTRIBUTE_ALWAYS_INLINE void eraseLastRow() {
    assert(Mat.idxFromRow(lastRow().data()).first == Mat.getNumRows() - 1 &&
           "Last row does not correspond to last row in storage");
    dropLastRow();
    Mat.eraseLastRow();
  }

protected:
  // Helper constructor.
  LLVM_ATTRIBUTE_ALWAYS_INLINE constexpr JaggedArrayView(
      MatrixStorage<T, NStorageInline> &Mat,
      SmallVectorImpl<row_type> &&RowView)
      : Mat(Mat), RowView(std::move(RowView)) {}

private:
  MatrixStorage<T, NStorageInline> &Mat;
  container_type RowView;
};
} // namespace llvm

#endif
