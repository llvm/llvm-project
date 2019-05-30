//==------------ accessor.hpp - SYCL standard header file ------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/atomic.hpp>
#include <CL/sycl/buffer.hpp>
#include <CL/sycl/detail/accessor_impl.hpp>
#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/handler.hpp>
#include <CL/sycl/id.hpp>
#include <CL/sycl/pointers.hpp>

// The file contains implementations of accessor class. Objects of accessor
// class define a requirement to access some SYCL memory object or local memory
// of the device.
//
// Basically there are 3 distinct types of accessors.
//
// One of them is an accessor to a SYCL buffer object(Buffer accessor) which has
// the richest interface. It supports things like accessing only a part of
// buffer, multidimensional access using sycl::id, conversions to various
// multi_ptr and atomic classes.
//
// Second type is an accessor to a SYCL image object(Image accessor) which has
// "image" specific methods for reading and writing.
//
// Finally, accessor to local memory(Local accessor) doesn't require access to
// any SYCL memory object, but asks for some local memory on device to be
// available. Some methods overlap with ones that "Buffer accessor" provides.
//
// Buffer and Image accessors create the requirement to access some SYCL memory
// object(or part of it). SYCL RT must detect when two kernels want to access
// the same memory objects and make sure they are executed in correct order.
//
// "accessor_common" class that contains several common methods between Buffer
// and Local accessors.
//
// Accessors have different representation on host and on device. On host they
// have non-templated base class, that is needed to safely work with any
// accessor type. Furhermore on host we need some additional fields in order
// to implement functionality required by Specification, for example during
// lifetime of a host accessor other operations with memory object the accessor
// refers to should be blocked and when all references to the host accessor are
// desctructed, the memory this host accessor refers to should be "written
// back".
//
// The scheme of inheritance for host side:
//
//  +------------------+     +-----------------+     +-----------------------+
//  |                  |     |                 |     |                       |
//  | AccessorBaseHost |     | accessor_common |     | LocalAccessorBaseHost |
//  |                  |     |                 |     |                       |
//  +------------------+     +-----+-----------+     +--------+--------------+
//         |     |                     |   |                   |
//         |     +-----------+    +----+   +---------+  +------+
//         |                 |    |                  |  |
//         v                 v    v                  v  v
//  +----------------+  +-----------------+   +-------------+
//  |                |  |   accessor(1)   |   | accessor(3) |
//  | image_accessor |  +-----------------|   +-------------+
//  |                |  | for targets:    |   | for target: |
//  +---+---+---+----+  |                 |   |             |
//      |   |   |       | host_buffer     |   | local       |
//      |   |   |       | global_buffer   |   +-------------+
//      |   |   |       | constant_buffer |
//      |   |   |       +-----------------+
//      |   |   |
//      |   |   +------------------------------------+
//      |   |                                        |
//      |   +----------------------+                 |
//      v                          v                 v
//  +-----------------+    +--------------+    +-------------+
//  |     acessor(2)  |    |  accessor(4) |    | accessor(5) |
//  +-----------------+    +--------------+    +-------------+
//  | for targets:    |    | for targets: |    | for target: |
//  |                 |    |              |    |             |
//  | host_image      |    |  image       |    | image_array |
//  +-----------------+    +--------------+    +-------------+
//
// For host side AccessorBaseHost/LocalAccessorBaseHost contains shared_ptr
// which points to AccessorImplHost/LocalAccessorImplHost object.
//
// The scheme of inheritance for device side:
//
//                            +-----------------+
//                            |                 |
//                            | accessor_common |
//                            |                 |
//                            +-----+-------+---+
//                                     |       |
//                                +----+       +-----+
//                                |                  |
//                                v                  v
//  +----------------+  +-----------------+   +-------------+
//  |                |  |   accessor(1)   |   | accessor(3) |
//  | image_accessor |  +-----------------|   +-------------+
//  |                |  | for targets:    |   | for target: |
//  +---+---+---+----+  |                 |   |             |
//      |   |   |       | host_buffer     |   | local       |
//      |   |   |       | global_buffer   |   +-------------+
//      |   |   |       | constant_buffer |
//      |   |   |       +-----------------+
//      |   |   |
//      |   |   +------------------------------------+
//      |   |                                        |
//      |   +----------------------+                 |
//      v                          v                 v
//  +-----------------+    +--------------+    +-------------+
//  |     acessor(2)  |    |  accessor(4) |    | accessor(5) |
//  +-----------------+    +--------------+    +-------------+
//  | for targets:    |    | for targets: |    | for target: |
//  |                 |    |              |    |             |
//  | host_image      |    |  image       |    | image_array |
//  +-----------------+    +--------------+    +-------------+
//
// For device side AccessorImplHost/LocalAccessorImplHost are fileds of
// accessor(1) and accessor(3).
//
// accessor(1) declares accessor as a template class and implements accessor
// class for access targets: host_buffer, global_buffer and constant_buffer.
//
// accessor(3) specializes accessor(1) for the local access target.
//
// image_accessor contains implements interfaces for access targets: host_image,
// image and image_array. But there are three distinct specializations of the
// accessor(1) (accessor(2), accessor(4), accessor(5)) that are just inherited
// from image_accessor.
//
// accessor_common contains several helpers common for both accessor(1) and
// accessor(3)

namespace cl {
namespace sycl {

template <typename DataT, int Dimensions, access::mode AccessMode,
          access::target AccessTarget = access::target::global_buffer,
          access::placeholder IsPlaceholder = access::placeholder::false_t>
class accessor;

namespace detail {

// The function extends or truncates number of dimensions of objects of id
// or ranges classes. When extending the new values are filled with
// DefaultValue, truncation just removes extra values.
template <int NewDim, int DefaultValue, template <int> class T, int OldDim>
static T<NewDim> convertToArrayOfN(T<OldDim> OldObj) {
  T<NewDim> NewObj;
  const int CopyDims = NewDim > OldDim ? OldDim : NewDim;
  for (int I = 0; I < CopyDims; ++I)
    NewObj[I] = OldObj[I];
  for (int I = CopyDims; I < NewDim; ++I)
    NewObj[I] = DefaultValue;
  return NewObj;
}

template <typename DataT, int Dimensions, access::mode AccessMode,
          access::target AccessTarget, access::placeholder IsPlaceholder>
class accessor_common {
protected:
  constexpr static bool IsPlaceH = IsPlaceholder == access::placeholder::true_t;
  constexpr static access::address_space AS = TargetToAS<AccessTarget>::AS;

  constexpr static bool IsHostBuf = AccessTarget == access::target::host_buffer;

  constexpr static bool IsGlobalBuf =
      AccessTarget == access::target::global_buffer;

  constexpr static bool IsConstantBuf =
      AccessTarget == access::target::constant_buffer;

  constexpr static bool IsAccessAnyWrite =
      AccessMode == access::mode::write ||
      AccessMode == access::mode::read_write ||
      AccessMode == access::mode::discard_write ||
      AccessMode == access::mode::discard_read_write;

  constexpr static bool IsAccessReadOnly = AccessMode == access::mode::read;

  constexpr static bool IsAccessReadWrite =
      AccessMode == access::mode::read_write;

  using RefType = typename detail::PtrValueType<DataT, AS>::type &;
  using PtrType = typename detail::PtrValueType<DataT, AS>::type *;

  using AccType =
      accessor<DataT, Dimensions, AccessMode, AccessTarget, IsPlaceholder>;

  // The class which allows to access value of N dimensional accessor using N
  // subscript operators, e.g. accessor[2][2][3]
  template <int SubDims> class AccessorSubscript {
    static constexpr int Dims = Dimensions;

    template <bool B, class T = void>
    using enable_if_t = typename std::enable_if<B, T>::type;

    mutable id<Dims> MIDs;
    AccType MAccessor;

  public:
    AccessorSubscript(AccType Accessor, id<Dims> IDs)
        : MAccessor(Accessor), MIDs(IDs) {}

    // Only accessor class is supposed to use this c'tor for the first
    // operator[].
    AccessorSubscript(AccType Accessor, size_t Index) : MAccessor(Accessor) {
      MIDs[0] = Index;
    }

    template <int CurDims = SubDims, typename = enable_if_t<(CurDims > 1)>>
    AccessorSubscript<CurDims - 1> operator[](size_t Index) {
      MIDs[Dims - CurDims] = Index;
      return AccessorSubscript<CurDims - 1>(MAccessor, MIDs);
    }

    template <int CurDims = SubDims,
              typename = enable_if_t<IsAccessAnyWrite && CurDims == 1>>
    RefType operator[](size_t Index) const {
      MIDs[Dims - CurDims] = Index;
      return MAccessor[MIDs];
    }

    template <int CurDims = SubDims,
              typename = enable_if_t<IsAccessReadOnly && CurDims == 1>>
    DataT operator[](size_t Index) const {
      MIDs[Dims - SubDims] = Index;
      return MAccessor[MIDs];
    }
  };
};

} // namespace detail

template <typename DataT, int Dimensions, access::mode AccessMode,
          access::target AccessTarget, access::placeholder IsPlaceholder>
class accessor :
#ifndef __SYCL_DEVICE_ONLY__
    public detail::AccessorBaseHost,
#endif
    public detail::accessor_common<DataT, Dimensions, AccessMode, AccessTarget,
                                   IsPlaceholder> {

  static_assert((AccessTarget == access::target::global_buffer ||
                 AccessTarget == access::target::constant_buffer ||
                 AccessTarget == access::target::host_buffer),
                "Expected buffer type");

  template <bool B, class T = void>
  using enable_if_t = typename std::enable_if<B, T>::type;

  using AccessorCommonT = detail::accessor_common<DataT, Dimensions, AccessMode,
                                                  AccessTarget, IsPlaceholder>;

  constexpr static int AdjustedDim = Dimensions == 0 ? 1 : Dimensions;

  using AccessorCommonT::AS;
  using AccessorCommonT::IsAccessAnyWrite;
  using AccessorCommonT::IsAccessReadOnly;
  using AccessorCommonT::IsConstantBuf;
  using AccessorCommonT::IsGlobalBuf;
  using AccessorCommonT::IsHostBuf;
  using AccessorCommonT::IsPlaceH;
  template <int Dims>
  using AccessorSubscript =
      typename AccessorCommonT::template AccessorSubscript<Dims>;

  using RefType = typename detail::PtrValueType<DataT, AS>::type &;
  using PtrType = typename detail::PtrValueType<DataT, AS>::type *;

  template <int Dims = Dimensions> size_t getLinearIndex(id<Dims> Id) const {

#ifdef __SYCL_DEVICE_ONLY__
    // Pointer is already adjusted for 1D case.
    if (Dimensions == 1)
      return Id[0];
#endif // __SYCL_DEVICE_ONLY__

    size_t Result = 0;
    for (int I = 0; I < Dims; ++I)
      Result = Result * getMemoryRange()[I] + getOffset()[I] + Id[I];
    return Result;
  }

#ifdef __SYCL_DEVICE_ONLY__

  id<AdjustedDim> &getOffset() { return impl.Offset; }
  range<AdjustedDim> &getAccessRange() { return impl.AccessRange; }
  range<AdjustedDim> &getMemoryRange() { return impl.MemRange; }

  const id<AdjustedDim> &getOffset() const { return impl.Offset; }
  const range<AdjustedDim> &getAccessRange() const { return impl.AccessRange; }
  const range<AdjustedDim> &getMemoryRange() const { return impl.MemRange; }

  detail::AccessorImplDevice<AdjustedDim> impl;

  PtrType MData;

  void __init(PtrType Ptr, range<AdjustedDim> AccessRange,
              range<AdjustedDim> MemRange, id<AdjustedDim> Offset) {
    MData = Ptr;
    for (int I = 0; I < AdjustedDim; ++I) {
      getOffset()[I] = Offset[I];
      getAccessRange()[I] = AccessRange[I];
      getMemoryRange()[I] = MemRange[I];
    }
    // In case of 1D buffer, adjust pointer during initialization rather
    // then each time in operator[] or get_pointer functions.
    if (1 == AdjustedDim)
      MData += Offset[0];
  }

  PtrType getQualifiedPtr() const { return MData; }
#else

  using AccessorBaseHost::getAccessRange;
  using AccessorBaseHost::getOffset;
  using AccessorBaseHost::getMemoryRange;

  char padding[sizeof(detail::AccessorImplDevice<AdjustedDim>) +
               sizeof(PtrType) - sizeof(detail::AccessorBaseHost)];

  PtrType getQualifiedPtr() const {
    return reinterpret_cast<PtrType>(AccessorBaseHost::getPtr());
  }

#endif // __SYCL_DEVICE_ONLY__

public:
  using value_type = DataT;
  using reference = DataT &;
  using const_reference = const DataT &;

  template <int Dims = Dimensions>
  accessor(enable_if_t<((!IsPlaceH && IsHostBuf) ||
                        (IsPlaceH && (IsGlobalBuf || IsConstantBuf))) &&
                       Dims == 0, buffer<DataT, 1> > &BufferRef)
#ifdef __SYCL_DEVICE_ONLY__
      : impl(id<AdjustedDim>(), BufferRef.get_range(), BufferRef.MemRange) {
#else
      : AccessorBaseHost(
            /*Offset=*/{0, 0, 0},
            detail::convertToArrayOfN<3, 1>(BufferRef.get_range()),
            detail::convertToArrayOfN<3, 1>(BufferRef.MemRange), AccessMode,
            detail::getSyclObjImpl(BufferRef).get(), AdjustedDim,
            sizeof(DataT)) {
    detail::EventImplPtr Event =
        detail::Scheduler::getInstance().addHostAccessor(
            AccessorBaseHost::impl.get());
    Event->wait(Event);
#endif
  }

  template <int Dims = Dimensions>
  accessor(
      buffer<DataT, 1> &BufferRef,
      enable_if_t<(!IsPlaceH && (IsGlobalBuf || IsConstantBuf)) && Dims == 0,
                   handler> &CommandGroupHandler)
#ifdef __SYCL_DEVICE_ONLY__
      : impl(id<AdjustedDim>(), BufferRef.get_range(), BufferRef.MemRange) {
  }
#else
      : AccessorBaseHost(
            /*Offset=*/{0, 0, 0},
            detail::convertToArrayOfN<3, 1>(BufferRef.get_range()),
            detail::convertToArrayOfN<3, 1>(BufferRef.MemRange), AccessMode,
            detail::getSyclObjImpl(BufferRef).get(), Dimensions,
            sizeof(DataT)) {
    CommandGroupHandler.set_arg(/*Index=*/0, *this);
  }
#endif

  template <
      int Dims = Dimensions,
      typename = enable_if_t<((!IsPlaceH && IsHostBuf) ||
                              (IsPlaceH && (IsGlobalBuf || IsConstantBuf))) &&
                             (Dims > 0)>>
  accessor(buffer<DataT, Dimensions> &BufferRef)
#ifdef __SYCL_DEVICE_ONLY__
      : impl(id<Dimensions>(), BufferRef.get_range(), BufferRef.MemRange) {
  }
#else
      : AccessorBaseHost(
            /*Offset=*/{0, 0, 0},
            detail::convertToArrayOfN<3, 1>(BufferRef.get_range()),
            detail::convertToArrayOfN<3, 1>(BufferRef.MemRange), AccessMode,
            detail::getSyclObjImpl(BufferRef).get(), Dimensions,
            sizeof(DataT)) {
    detail::EventImplPtr Event =
        detail::Scheduler::getInstance().addHostAccessor(
            AccessorBaseHost::impl.get());
    Event->wait(Event);
  }
#endif

  template <int Dims = Dimensions,
            typename = enable_if_t<
                (!IsPlaceH && (IsGlobalBuf || IsConstantBuf)) && (Dims > 0)>>
  accessor(buffer<DataT, Dimensions> &BufferRef, handler &CommandGroupHandler)
#ifdef __SYCL_DEVICE_ONLY__
      : impl(id<AdjustedDim>(), BufferRef.get_range(), BufferRef.MemRange) {
  }
#else
      : AccessorBaseHost(
            /*Offset=*/{0, 0, 0},
            detail::convertToArrayOfN<3, 1>(BufferRef.get_range()),
            detail::convertToArrayOfN<3, 1>(BufferRef.MemRange), AccessMode,
            detail::getSyclObjImpl(BufferRef).get(), Dimensions,
            sizeof(DataT)) {
    CommandGroupHandler.set_arg(/*Index=*/0, *this);
  }
#endif

  template <
      int Dims = Dimensions,
      typename = enable_if_t<((!IsPlaceH && IsHostBuf) ||
                              (IsPlaceH && (IsGlobalBuf || IsConstantBuf))) &&
                             (Dims > 0)>>
  accessor(buffer<DataT, Dimensions> &BufferRef, range<Dimensions> AccessRange,
           id<Dimensions> AccessOffset = {})
#ifdef __SYCL_DEVICE_ONLY__
      : impl(AccessOffset, AccessRange, BufferRef.MemRange) {
  }
#else
      : AccessorBaseHost(detail::convertToArrayOfN<3, 0>(AccessOffset),
                           detail::convertToArrayOfN<3, 1>(AccessRange),
                           detail::convertToArrayOfN<3, 1>(BufferRef.MemRange),
                           AccessMode, detail::getSyclObjImpl(BufferRef).get(),
                           Dimensions, sizeof(DataT)) {
    detail::EventImplPtr Event =
        detail::Scheduler::getInstance().addHostAccessor(
            AccessorBaseHost::impl.get());
    Event->wait(Event);
  }
#endif

  template <int Dims = Dimensions,
            typename = enable_if_t<
                (!IsPlaceH && (IsGlobalBuf || IsConstantBuf)) && (Dims > 0)>>
  accessor(buffer<DataT, Dimensions> &BufferRef, handler &CommandGroupHandler,
           range<Dimensions> AccessRange, id<Dimensions> AccessOffset = {})
#ifdef __SYCL_DEVICE_ONLY__
      : impl(AccessOffset, AccessRange, BufferRef.MemRange) {
  }
#else
      : AccessorBaseHost(detail::convertToArrayOfN<3, 0>(AccessOffset),
                           detail::convertToArrayOfN<3, 1>(AccessRange),
                           detail::convertToArrayOfN<3, 1>(BufferRef.MemRange),
                           AccessMode, detail::getSyclObjImpl(BufferRef).get(),
                           Dimensions, sizeof(DataT)) {
    CommandGroupHandler.set_arg(/*Index=*/0, *this);
  }
#endif

  constexpr bool is_placeholder() const { return IsPlaceH; }

  size_t get_size() const { return getMemoryRange().size() * sizeof(DataT); }

  size_t get_count() const { return getMemoryRange().size(); }

  template <int Dims = Dimensions, typename = enable_if_t<(Dims > 0)>>
  range<Dimensions> get_range() const {
    return detail::convertToArrayOfN<Dimensions, 1>(getAccessRange());
  }

  template <int Dims = Dimensions, typename = enable_if_t<(Dims > 0)>>
  id<Dimensions> get_offset() const {
    return detail::convertToArrayOfN<Dimensions, 0>(getOffset());
  }

  template <int Dims = Dimensions,
            typename = enable_if_t<IsAccessAnyWrite && Dims == 0>>
  operator RefType() const {
    const size_t LinearIndex = getLinearIndex(id<AdjustedDim>());
    return *(getQualifiedPtr() + LinearIndex);
  }

  template <int Dims = Dimensions,
            typename = enable_if_t<IsAccessAnyWrite && (Dims > 0)>>
  RefType operator[](id<Dimensions> Index) const {
    const size_t LinearIndex = getLinearIndex(Index);
    return getQualifiedPtr()[LinearIndex];
  }

  template <int Dims = Dimensions,
            typename = enable_if_t<IsAccessAnyWrite && Dims == 1>>
  RefType operator[](size_t Index) const {
    const size_t LinearIndex = getLinearIndex(id<Dimensions>(Index));
    return getQualifiedPtr()[LinearIndex];
  }

  template <int Dims = Dimensions,
            typename = enable_if_t<IsAccessReadOnly && Dims == 0>>
  operator DataT() const {
    const size_t LinearIndex = getLinearIndex(id<AdjustedDim>());
    return *(getQualifiedPtr() + LinearIndex);
  }

  template <int Dims = Dimensions,
            typename = enable_if_t<IsAccessReadOnly && (Dims > 0)>>
  DataT operator[](id<Dimensions> Index) const {
    const size_t LinearIndex = getLinearIndex(Index);
    return getQualifiedPtr()[LinearIndex];
  }

  template <int Dims = Dimensions,
            typename = enable_if_t<IsAccessReadOnly && Dims == 1>>
  DataT operator[](size_t Index) const {
    const size_t LinearIndex = getLinearIndex(id<Dimensions>(Index));
    return getQualifiedPtr()[LinearIndex];
  }

  template <
      int Dims = Dimensions,
      typename = enable_if_t<AccessMode == access::mode::atomic && Dims == 0>>
  operator atomic<DataT, AS>() const {
    const size_t LinearIndex = getLinearIndex(id<AdjustedDim>());
    return atomic<DataT, AS>(
        multi_ptr<DataT, AS>(getQualifiedPtr() + LinearIndex));
  }

  template <
      int Dims = Dimensions,
      typename = enable_if_t<AccessMode == access::mode::atomic && (Dims > 0)>>
  atomic<DataT, AS> operator[](id<Dimensions> Index) const {
    const size_t LinearIndex = getLinearIndex(Index);
    return atomic<DataT, AS>(
        multi_ptr<DataT, AS>(getQualifiedPtr() + LinearIndex));
  }

  template <
      int Dims = Dimensions,
      typename = enable_if_t<AccessMode == access::mode::atomic && Dims == 1>>
  atomic<DataT, AS> operator[](size_t Index) const {
    const size_t LinearIndex = getLinearIndex(id<AdjustedDim>(Index));
    return atomic<DataT, AS>(
        multi_ptr<DataT, AS>(getQualifiedPtr() + LinearIndex));
  }

  template <int Dims = Dimensions, typename = enable_if_t<(Dims > 1)>>
  typename AccessorCommonT::template AccessorSubscript<Dims - 1>
  operator[](size_t Index) const {
    return AccessorSubscript<Dims - 1>(*this, Index);
  }

  template <
      access::target AccessTarget_ = AccessTarget,
      typename = enable_if_t<AccessTarget_ == access::target::host_buffer>>
  DataT *get_pointer() const {
    const size_t LinearIndex = getLinearIndex(id<AdjustedDim>());
    return getQualifiedPtr() + LinearIndex;
  }

  template <
      access::target AccessTarget_ = AccessTarget,
      typename = enable_if_t<AccessTarget_ == access::target::global_buffer>>
  global_ptr<DataT> get_pointer() const {
    const size_t LinearIndex = getLinearIndex(id<AdjustedDim>());
    return global_ptr<DataT>(getQualifiedPtr() + LinearIndex);
  }

  template <
      access::target AccessTarget_ = AccessTarget,
      typename = enable_if_t<AccessTarget_ == access::target::constant_buffer>>
  constant_ptr<DataT> get_pointer() const {
    const size_t LinearIndex = getLinearIndex(id<AdjustedDim>());
    return constant_ptr<DataT>(getQualifiedPtr() + LinearIndex);
  }

  bool operator==(const accessor &Rhs) const { return impl == Rhs.impl; }
  bool operator!=(const accessor &Rhs) const { return !(*this == Rhs); }
};

// Local accessor
template <typename DataT, int Dimensions, access::mode AccessMode,
          access::placeholder IsPlaceholder>
class accessor<DataT, Dimensions, AccessMode, access::target::local,
               IsPlaceholder> :
#ifndef __SYCL_DEVICE_ONLY__
    public detail::LocalAccessorBaseHost,
#endif
    public detail::accessor_common<DataT, Dimensions, AccessMode,
                                   access::target::local, IsPlaceholder> {

  constexpr static int AdjustedDim = Dimensions == 0 ? 1 : Dimensions;

  using AccessorCommonT =
      detail::accessor_common<DataT, Dimensions, AccessMode,
                              access::target::local, IsPlaceholder>;

  using AccessorCommonT::AS;
  using AccessorCommonT::IsAccessAnyWrite;
  template <int Dims>
  using AccessorSubscript =
      typename AccessorCommonT::template AccessorSubscript<Dims>;

  using RefType = typename detail::PtrValueType<DataT, AS>::type &;
  using PtrType = typename detail::PtrValueType<DataT, AS>::type *;

  template <bool B, class T = void>
  using enable_if_t = typename std::enable_if<B, T>::type;

#ifdef __SYCL_DEVICE_ONLY__
  detail::LocalAccessorBaseDevice<AdjustedDim> impl;

  sycl::range<AdjustedDim> &getSize() { return impl.MemRange; }
  const sycl::range<AdjustedDim> &getSize() const { return impl.MemRange; }

  void __init(PtrType Ptr, range<AdjustedDim> AccessRange,
              range<AdjustedDim> MemRange, id<AdjustedDim> Offset) {
    MData = Ptr;
    for (int I = 0; I < AdjustedDim; ++I)
      getSize()[I] = AccessRange[I];
  }

  PtrType getQualifiedPtr() const { return MData; }

  PtrType MData;

#else

  char padding[sizeof(detail::LocalAccessorBaseDevice<AdjustedDim>) +
               sizeof(PtrType) - sizeof(detail::LocalAccessorBaseHost)];
  using detail::LocalAccessorBaseHost::getSize;

  PtrType getQualifiedPtr() const {
    return reinterpret_cast<PtrType>(LocalAccessorBaseHost::getPtr());
  }

#endif // __SYCL_DEVICE_ONLY__

  // Method which calculates linear offset for the ID using Range and Offset.
  template <int Dims = AdjustedDim> size_t getLinearIndex(id<Dims> Id) const {
    size_t Result = 0;
    for (int I = 0; I < Dims; ++I)
      Result = Result * getSize()[I] + Id[I];
    return Result;
  }

public:
  using value_type = DataT;
  using reference = DataT &;
  using const_reference = const DataT &;

  template <int Dims = Dimensions, typename = enable_if_t<Dims == 0>>
  accessor(handler &CommandGroupHandler)
#ifdef __SYCL_DEVICE_ONLY__
      : impl(range<AdjustedDim>{1}) {
  }
#else
      : LocalAccessorBaseHost(range<3>{1, 1, 1}, AdjustedDim, sizeof(DataT)) {
  }
#endif

  template <int Dims = Dimensions, typename = enable_if_t<(Dims > 0)>>
  accessor(range<Dimensions> AllocationSize, handler &CommandGroupHandler)
#ifdef __SYCL_DEVICE_ONLY__
      : impl(AllocationSize) {
  }
#else
      : LocalAccessorBaseHost(detail::convertToArrayOfN<3, 1>(AllocationSize),
                              AdjustedDim, sizeof(DataT)) {
  }
#endif

  size_t get_size() const { return getSize().size() * sizeof(DataT); }

  size_t get_count() const { return getSize().size(); }

  template <int Dims = Dimensions,
            typename = enable_if_t<IsAccessAnyWrite && Dims == 0>>
  operator RefType() const {
    return *getQualifiedPtr();
  }

  template <int Dims = Dimensions,
            typename = enable_if_t<IsAccessAnyWrite && (Dims > 0)>>
  RefType operator[](id<Dimensions> Index) const {
    const size_t LinearIndex = getLinearIndex(Index);
    return getQualifiedPtr()[LinearIndex];
  }

  template <int Dims = Dimensions,
            typename = enable_if_t<IsAccessAnyWrite && Dims == 1>>
  RefType operator[](size_t Index) const {
    return getQualifiedPtr()[Index];
  }

  template <
      int Dims = Dimensions,
      typename = enable_if_t<AccessMode == access::mode::atomic && Dims == 0>>
  operator atomic<DataT, AS>() const {
    return atomic<DataT, AS>(multi_ptr<DataT, AS>(getQualifiedPtr()));
  }

  template <
      int Dims = Dimensions,
      typename = enable_if_t<AccessMode == access::mode::atomic && (Dims > 0)>>
  atomic<DataT, AS> operator[](id<Dimensions> Index) const {
    const size_t LinearIndex = getLinearIndex(Index);
    return atomic<DataT, AS>(
        multi_ptr<DataT, AS>(getQualifiedPtr() + LinearIndex));
  }

  template <
      int Dims = Dimensions,
      typename = enable_if_t<AccessMode == access::mode::atomic && Dims == 1>>
  atomic<DataT, AS> operator[](size_t Index) const {
    return atomic<DataT, AS>(multi_ptr<DataT, AS>(getQualifiedPtr() + Index));
  }

  template <int Dims = Dimensions, typename = enable_if_t<(Dims > 1)>>
  typename AccessorCommonT::template AccessorSubscript<Dims - 1>
  operator[](size_t Index) const {
    return AccessorSubscript<Dims - 1>(*this, Index);
  }

  local_ptr<DataT> get_pointer() const {
    return local_ptr<DataT>(getQualifiedPtr());
  }

  bool operator==(const accessor &Rhs) const { return impl == Rhs.impl; }
  bool operator!=(const accessor &Rhs) const { return !(*this == Rhs); }
};

// Image accessor
template <typename DataT, int Dimensions, access::mode AccessMode,
          access::target AccessTarget, access::placeholder IsPlaceholder>
class image_accessor {
  static_assert(AccessTarget == access::target::image ||
                    AccessTarget == access::target::host_image ||
                    AccessTarget == access::target::image_array,
                "Expected image type");
  // TODO: Check if placeholder is applicable here.
public:
  using value_type = DataT;
  using reference = DataT &;
  using const_reference = const DataT &;

  /* Available only when: accessTarget == access::target::host_image */
  // template <typename AllocatorT>
  // accessor(image<dimensions, AllocatorT> &imageRef);
  /* Available only when: accessTarget == access::target::image */
  // template <typename AllocatorT>
  // accessor(image<dimensions, AllocatorT> &imageRef,
  // handler &commandGroupHandlerRef);

  /* Available only when: accessTarget == access::target::image_array &&
        dimensions < 3 */
  // template <typename AllocatorT>
  // accessor(image<dimensions + 1, AllocatorT> &imageRef,
  // handler &commandGroupHandlerRef);

  /* TODO -- common interface members -- */
  // size_t get_size() const;

  // size_t get_count() const;

  /* Available only when: (accessTarget == access::target::image || accessTarget
     == access::target::host_image) && accessMode == access::mode::read */
  // template <typename coordT> dataT read(const coordT &coords) const;

  /* Available only when: (accessTarget == access::target::image || accessTarget
     == access::target::host_image) && accessMode == access::mode::read */
  // template <typename coordT>
  // dataT read(const coordT &coords, const sampler &smpl) const;

  /* Available only when: (accessTarget == access::target::image || accessTarget
      == access::target::host_image) && accessMode == access::mode::write ||
      accessMode == access::mode::discard_write */
  // template <typename coordT>
  // void write(const coordT &coords, const dataT &color) const;

  /* Available only when: accessTarget == access::target::image_array &&
     dimensions < 3 */
  //__image_array_slice__ operator[](size_t index) const;
};

// Image accessors
template <typename DataT, int Dimensions, access::mode AccessMode,
          access::placeholder IsPlaceholder>
class accessor<DataT, Dimensions, AccessMode, access::target::image,
               IsPlaceholder>
    : public image_accessor<DataT, Dimensions, AccessMode,
                            access::target::image, IsPlaceholder> {};

template <typename DataT, int Dimensions, access::mode AccessMode,
          access::placeholder IsPlaceholder>
class accessor<DataT, Dimensions, AccessMode, access::target::host_image,
               IsPlaceholder>
    : public image_accessor<DataT, Dimensions, AccessMode,
                            access::target::host_image, IsPlaceholder> {};

template <typename DataT, int Dimensions, access::mode AccessMode,
          access::placeholder IsPlaceholder>
class accessor<DataT, Dimensions, AccessMode, access::target::image_array,
               IsPlaceholder>
    : public image_accessor<DataT, Dimensions, AccessMode,
                            access::target::image_array, IsPlaceholder> {};

} // namespace sycl
} // namespace cl



namespace std {
template <typename DataT, int Dimensions, cl::sycl::access::mode AccessMode,
          cl::sycl::access::target AccessTarget,
          cl::sycl::access::placeholder IsPlaceholder>
struct hash<cl::sycl::accessor<DataT, Dimensions, AccessMode, AccessTarget,
                               IsPlaceholder>> {
  using AccType = cl::sycl::accessor<DataT, Dimensions, AccessMode,
                                     AccessTarget, IsPlaceholder>;

  size_t operator()(const AccType &A) const {
#ifdef __SYCL_DEVICE_ONLY__
    // Hash is not supported on DEVICE. Just return 0 here.
    return 0;
#else
    // getSyclObjImpl() here returns a pointer to either AccessorImplHost
    // or LocalAccessorImplHost depending on the AccessTarget.
    auto AccImplPtr = cl::sycl::detail::getSyclObjImpl(A);
    return hash<decltype(AccImplPtr)>()(AccImplPtr);
#endif
  }
};

} // namespace std
