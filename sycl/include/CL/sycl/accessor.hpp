//==--------- accessor.hpp --- SYCL accessor -------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <type_traits>

#include <CL/sycl/atomic.hpp>
#include <CL/sycl/buffer.hpp>
#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/handler.hpp>
#include <CL/sycl/id.hpp>
#include <CL/sycl/pointers.hpp>
#include <CL/sycl/detail/scheduler/scheduler.h>
#include <CL/sycl/stl.hpp>

namespace cl {
namespace sycl {
namespace detail {

template <typename dataT, int dimensions, access::mode accessMode,
          access::target accessTarget, access::placeholder isPlaceholder>
class accessor_base;

template <int accessorDim, typename dataT, int dimensions,
          access::mode accessMode, access::target accessTarget,
          access::placeholder isPlaceholder>
class subscript_obj {
  using accessor_t = accessor_base<dataT, accessorDim, accessMode, accessTarget,
                                   isPlaceholder>;

  // TODO: Remove reference here as subscript_obj, can potentially outlive
  // the accessor. There is no spec-defined usecase, so leave it for now.
  const accessor_t &accRef;
  cl::sycl::id<accessorDim> ids;

public:
  subscript_obj(const accessor_t &acc, cl::sycl::id<accessorDim> &indexes)
      : accRef(acc), ids(indexes) {}

  subscript_obj<accessorDim, dataT, dimensions - 1, accessMode, accessTarget,
                isPlaceholder>
  operator[](size_t index) {
    ids[accessorDim - dimensions] = index;
    return subscript_obj<accessorDim, dataT, dimensions - 1, accessMode,
                         accessTarget, isPlaceholder>(accRef, ids);
  }
};

template <int accessorDim, typename dataT, access::mode accessMode,
          access::target accessTarget, access::placeholder isPlaceholder>
class subscript_obj<accessorDim, dataT, 1, accessMode, accessTarget,
                    isPlaceholder> {
  using accessor_t = accessor_base<dataT, accessorDim, accessMode, accessTarget,
                                   isPlaceholder>;

  const accessor_t &accRef;
  cl::sycl::id<accessorDim> ids;

public:
  subscript_obj(const accessor_t &acc, cl::sycl::id<accessorDim> &indexes)
      : accRef(acc), ids(indexes) {}

  dataT &operator[](size_t index) {
    ids[accessorDim - 1] = index;
    return accRef.__get_impl()->Data[getOffsetForId(
      accRef.__get_impl()->MemRange, ids, accRef.__get_impl()->Offset)];
  }
};

template <int accessorDim, typename dataT,
          access::target accessTarget, access::placeholder isPlaceholder>
class subscript_obj<accessorDim, dataT, 1, access::mode::read, accessTarget,
                    isPlaceholder> {
  using accessor_t = accessor_base<dataT, accessorDim, access::mode::read,
                                   accessTarget, isPlaceholder>;

  const accessor_t &accRef;
  cl::sycl::id<accessorDim> ids;

public:
  subscript_obj(const accessor_t &acc, cl::sycl::id<accessorDim> &indexes)
      : accRef(acc), ids(indexes) {}

  typename detail::remove_AS<dataT>::type
  operator[](size_t index) {
    ids[accessorDim - 1] = index;
    return accRef.__get_impl()->Data[getOffsetForId(
      accRef.__get_impl()->MemRange, ids, accRef.__get_impl()->Offset)];
  }
};

/// Specializations of accessor_impl define data fields for accessor.
/// There is no default implementation for the class. This class is
/// not a root of the class hierarchy, because it should be
/// initialized at the bottom of the hierarchy.
template <typename dataT, int dimensions, access::mode accessMode,
          access::target accessTarget, access::placeholder isPlaceholder,
          typename voidT = void>
struct accessor_impl;

#define SYCL_ACCESSOR_IMPL(CONDITION)                                          \
  template <typename dataT, int dimensions, access::mode accessMode,           \
            access::target accessTarget, access::placeholder isPlaceholder>    \
  struct accessor_impl<dataT, dimensions, accessMode, accessTarget,            \
                       isPlaceholder,                                          \
                       typename std::enable_if<(CONDITION)>::type>

/// Implementation of host accessor providing access to a single element.
/// Available when (dimensions == 0).
SYCL_ACCESSOR_IMPL(isTargetHostAccess(accessTarget) && dimensions == 0) {
  dataT *Data;
  // For simplicity of the padding in the main accessor class
  // we ensure that accessor_impl is not smaller than size of shared_ptr.
  char padding[sizeof(std::shared_ptr<accessor_impl>) - sizeof(dataT *)];
  accessor_impl(dataT *Data) : Data(Data) {}

  // Returns the number of accessed elements.
  size_t get_count() const { return 1; }

#ifdef __SYCL_DEVICE_ONLY__
  bool operator==(const accessor_impl &Rhs) const { return Data == Rhs.Data; }
#endif
};

/// Implementation of host accessor.
/// Available when (dimensions > 0).
SYCL_ACCESSOR_IMPL(isTargetHostAccess(accessTarget) && dimensions > 0) {
  dataT *Data;
  // Accessor's own range, can be subset of MemRange
  range<dimensions> AccessRange;
  // Range of corresponding memory object
  range<dimensions> MemRange;
  id<dimensions> Offset;

  accessor_impl(dataT * Data, range<dimensions> AccessRange,
                range<dimensions> MemRange, id<dimensions> Offset = {})
      : Data(Data), AccessRange(AccessRange), MemRange(MemRange),
        Offset(Offset) {}

  // Returns the number of accessed elements.
  size_t get_count() const { return AccessRange.size(); }

#ifdef __SYCL_DEVICE_ONLY__
  bool operator==(const accessor_impl &Rhs) const {
    return Data == Rhs.Data && AccessRange == Rhs.AccessRange &&
           MemRange == Rhs.MemRange && Offset == Rhs.Offset;
  }
#endif
};

/// Implementation of device (kernel) accessor providing access to a single
/// element. Available only when (dimensions == 0).
/// There is no way to tell at compile time if this accessor will be used
/// on OpenCL device or on host. So, the class should fit both variants.
SYCL_ACCESSOR_IMPL(!isTargetHostAccess(accessTarget) &&
                   accessTarget != access::target::local &&
                   dimensions == 0) {
  // This field must be the first to guarantee that it's safe to use
  // reinterpret casting while setting kernel arguments in order to get cl_mem
  // value from the buffer regardless of the accessor's dimensionality.
#ifndef __SYCL_DEVICE_ONLY__
  detail::buffer_impl<buffer_allocator> *m_Buf = nullptr;
#else
  char padding[sizeof(detail::buffer_impl<buffer_allocator> *)];
#endif // __SYCL_DEVICE_ONLY__

  dataT *Data;

  // Device accessors must be associated with a command group handler.
  // The handler though can be nullptr at the creation point if the
  // accessor is a placeholder accessor.
  accessor_impl(dataT *Data, handler *Handler = nullptr)
      : Data(Data)
  {}

  // Returns the number of accessed elements.
  size_t get_count() const { return 1; }

#ifdef __SYCL_DEVICE_ONLY__
  bool operator==(const accessor_impl &Rhs) const { return Data == Rhs.Data; }
#endif

  static_assert(
      std::is_same<typename DeviceValueType<dataT, accessTarget>::type,
                   dataT>::value,
      "The type should have been adjusted before propagating through "
      "class hierarchy");
};

/// Implementation of device (kernel) accessor. There is no way to
/// tell at compile time if this accessor will be used on OpenCL
/// device or on host. So, the class should fit both variants.
/// Available only when (dimensions > 0).
SYCL_ACCESSOR_IMPL(!isTargetHostAccess(accessTarget) &&
                   accessTarget != access::target::local &&
                   dimensions > 0) {
  // This field must be the first to guarantee that it's safe to use
  // reinterpret casting while setting kernel arguments in order to get cl_mem
  // value from the buffer regardless of the accessor's dimensionality.
#ifndef __SYCL_DEVICE_ONLY__
  detail::buffer_impl<buffer_allocator> *m_Buf = nullptr;
#else
  char padding[sizeof(detail::buffer_impl<buffer_allocator> *)];
#endif // __SYCL_DEVICE_ONLY__

  dataT *Data;
  // Accessor's own range, can be subset of MemRange
  range<dimensions> AccessRange;
  // Range of corresponding memory object
  range<dimensions> MemRange;
  id<dimensions> Offset;

  // Device accessors must be associated with a command group handler.
  // The handler though can be nullptr at the creation point if the
  // accessor is a placeholder accessor.
  accessor_impl(dataT * Data, range<dimensions> AccessRange,
                range<dimensions> MemRange, handler *Handler = nullptr,
                id<dimensions> Offset = {})
      : Data(Data), AccessRange(AccessRange), MemRange(MemRange),
        Offset(Offset) {}

  // Returns the number of accessed elements.
  size_t get_count() const { return AccessRange.size(); }

#ifdef __SYCL_DEVICE_ONLY__
  bool operator==(const accessor_impl &Rhs) const {
    return Data == Rhs.Data && AccessRange == Rhs.AccessRange &&
           MemRange == Rhs.MemRange && Offset == Rhs.Offset;
  }
#endif

  static_assert(
      std::is_same<typename DeviceValueType<dataT, accessTarget>::type,
                   dataT>::value,
      "The type should have been adjusted before propagating through "
      "class hierarchy");
};

/// Implementation of local accessor providing access to a single element.
/// Available only when (dimensions == 0).
SYCL_ACCESSOR_IMPL(accessTarget == access::target::local &&
                   dimensions == 0) {
  // This field must be the first to guarantee that it's safe to use
  // reinterpret casting while setting kernel arguments in order to get size
  // value from the accessor regardless of its dimensionality.
  size_t ByteSize;

#ifndef __SYCL_DEVICE_ONLY__
  shared_ptr_class<vector_class<dataT>> dataBuf;
#else
  char padding[sizeof(shared_ptr_class<vector_class<dataT>>)];
#endif

  dataT *Data;

  accessor_impl(handler * Handler)
      : ByteSize(sizeof(dataT))
  {
#ifndef __SYCL_DEVICE_ONLY__
    assert(Handler != nullptr && "Handler is nullptr");
    if (Handler->is_host()) {
      dataBuf = std::make_shared<vector_class<dataT>>(1);
      Data = dataBuf->data();
    }
#endif
  }

  // Returns the number of accessed elements.
  size_t get_count() const { return 1; }

#ifdef __SYCL_DEVICE_ONLY__
  bool operator==(const accessor_impl &Rhs) const {
    return ByteSize == Rhs.ByteSize && Data == Rhs.Data;
  }
#endif

  static_assert(
      std::is_same<typename DeviceValueType<dataT, accessTarget>::type,
                   dataT>::value,
      "The type should have been adjusted before propagating through "
      "class hierarchy");
};

/// Implementation of local accessor.
/// Available only when (dimensions > 0).
SYCL_ACCESSOR_IMPL(accessTarget == access::target::local &&
                   dimensions > 0) {
  // This field must be the first to guarantee that it's safe to use
  // reinterpret casting while setting kernel arguments in order to get size
  // value from the accessor regardless of its dimensionality.
  size_t ByteSize;

#ifndef __SYCL_DEVICE_ONLY__
  shared_ptr_class<vector_class<dataT>> dataBuf;
#else
  char padding[sizeof(shared_ptr_class<vector_class<dataT>>)];
#endif

  dataT *Data;
  // Accessor's own range
  range<dimensions> AccessRange;
  // For local accessor AccessRange and MemRange always are same but both fields
  // are presented here to keep no differences between local and global
  // accessors for compiler
  range<dimensions> MemRange;
  // TODO delete it when accessor class was remade
  // Offset field is not need for local accessor, but this field is now used
  // in the inheritance hierarchy. Getting rid of this field will cause
  // duplication and complication of the code even more.
  id<dimensions> Offset;

  accessor_impl(range<dimensions> Range, handler * Handler)
      : AccessRange(Range), MemRange(Range),
        ByteSize(Range.size() * sizeof(dataT)) {
#ifndef __SYCL_DEVICE_ONLY__
    assert(Handler != nullptr && "Handler is nullptr");
    if (Handler->is_host()) {
      dataBuf = std::make_shared<vector_class<dataT>>(Range.size());
      Data = dataBuf->data();
    }
#endif
  }

  // Returns the number of accessed elements.
  size_t get_count() const { return AccessRange.size(); }

#ifdef __SYCL_DEVICE_ONLY__
  bool operator==(const accessor_impl &Rhs) const {
    return ByteSize == Rhs.ByteSize && Data == Rhs.Data &&
           AccessRange == Rhs.AccessRange && MemRange == Rhs.MemRange &&
           Offset == Rhs.Offset;
  }
#endif

  static_assert(
      std::is_same<typename DeviceValueType<dataT, accessTarget>::type,
                   dataT>::value,
      "The type should have been adjusted before propagating through "
      "class hierarchy");
};

/// Base class for all accessor specializations.
template <typename dataT, int dimensions, access::mode accessMode,
          access::target accessTarget, access::placeholder isPlaceholder>
class accessor_base {
protected:
  template <int, typename, int, access::mode, access::target,
            access::placeholder>
  friend class subscript_obj;
  friend class ::cl::sycl::simple_scheduler::Node;
  friend class ::cl::sycl::simple_scheduler::Scheduler;
  using _ImplT =
      accessor_impl<dataT, dimensions, accessMode, accessTarget, isPlaceholder>;

  const _ImplT *__get_impl() const {
#ifdef __SYCL_DEVICE_ONLY__
    return reinterpret_cast<const _ImplT *>(this);
#else
    auto ImplPtrPtr = reinterpret_cast<const std::shared_ptr<_ImplT> *>(this);
    const _ImplT* I = &**ImplPtrPtr;
    return I;
#endif
  }

  _ImplT *__get_impl() {
#ifdef __SYCL_DEVICE_ONLY__
    return reinterpret_cast<_ImplT *>(this);
#else
    auto ImplPtrPtr = reinterpret_cast<std::shared_ptr<_ImplT> *>(this);
    _ImplT* I = &**ImplPtrPtr;
    return I;
#endif
  }

  static_assert(
      std::is_same<typename DeviceValueType<dataT, accessTarget>::type,
                   dataT>::value,
      "The type should have been adjusted before propagating through "
      "class hierarchy");
};

// The macro is used to conditionally define methods of accessor class
// by wrapping them into a structure that is non-empty only if the
// condition is met.
#define SYCL_ACCESSOR_SUBCLASS(TAG, PARENT, CONDITION)                         \
  template <typename dataT, int dimensions, access::mode accessMode,           \
            access::target accessTarget, access::placeholder isPlaceholder,    \
            typename voidT = void>                                             \
  struct TAG : ::cl::sycl::detail::PARENT<dataT, dimensions, accessMode,       \
                                          accessTarget, isPlaceholder> {};     \
                                                                               \
  template <typename dataT, int dimensions, access::mode accessMode,           \
            access::target accessTarget, access::placeholder isPlaceholder>    \
  struct TAG<dataT, dimensions, accessMode, accessTarget, isPlaceholder,       \
             typename std::enable_if<(CONDITION)>::type>                       \
      : ::cl::sycl::detail::PARENT<dataT, dimensions, accessMode,              \
                                   accessTarget, isPlaceholder>

SYCL_ACCESSOR_SUBCLASS(accessor_common, accessor_base, true /* always */) {
  // Returns true if the current accessor is a placeholder accessor.
  constexpr bool is_placeholder() const {
    return isPlaceholder == access::placeholder::true_t;
  }

  // Returns the size of the accessed memory in bytes.
  size_t get_size() const { return this->get_count() * sizeof(dataT); }

  // Returns the number of accessed elements.
  size_t get_count() const { return this->__get_impl()->get_count(); }

  template <int Dimensions = dimensions>
  typename std::enable_if<(Dimensions > 0), range<Dimensions>>::type
  get_range() const { return this->__get_impl()->AccessRange; }

  template <int Dimensions = dimensions>
  typename std::enable_if<(Dimensions > 0), id<Dimensions>>::type
  get_offset() const { return this->__get_impl()->Offset; }
};

SYCL_ACCESSOR_SUBCLASS(accessor_opdata_w, accessor_common,
                       (accessMode == access::mode::write ||
                        accessMode == access::mode::read_write ||
                        accessMode == access::mode::discard_write ||
                        accessMode == access::mode::discard_read_write) &&
                       dimensions == 0) {
  operator dataT &() const {
    return this->__get_impl()->Data[0];
  }
};

SYCL_ACCESSOR_SUBCLASS(accessor_subscript_wn, accessor_opdata_w,
                       (accessMode == access::mode::write ||
                        accessMode == access::mode::read_write ||
                        accessMode == access::mode::discard_write ||
                        accessMode == access::mode::discard_read_write) &&
                       dimensions > 0) {
  dataT &operator[](id<dimensions> index) const {
    return this->__get_impl()->Data[getOffsetForId(
      this->__get_impl()->MemRange, index, this->get_offset())];
  }

  subscript_obj<dimensions, dataT, dimensions - 1, accessMode, accessTarget,
              isPlaceholder>
  operator[](size_t index) const {
    id<dimensions> ids;
    ids[0] = index;
    return subscript_obj<dimensions, dataT, dimensions - 1, accessMode,
                         accessTarget, isPlaceholder>(*this, ids);
  }
};

SYCL_ACCESSOR_SUBCLASS(accessor_subscript_w, accessor_subscript_wn,
                       (accessMode == access::mode::write ||
                        accessMode == access::mode::read_write ||
                        accessMode == access::mode::discard_write ||
                        accessMode == access::mode::discard_read_write) &&
                       dimensions == 1) {
  // The tricky part here is that there is no function overloading
  // between different scopes in C++. That is, operator[] defined in a
  // child class hides any operator[] defined in any of the parent
  // classes. That's why operator[] defined in accessor_subscript_wn
  // is not visible here and we have to define
  // operator[](id<dimensions>) once again.
  dataT &operator[](id<dimensions> index) const {
    return this->operator[](
      getOffsetForId(this->__get_impl()->MemRange, index, this->get_offset()));
  }
  dataT &operator[](size_t index) const {
    return this->__get_impl()->Data[index];
  }
};

SYCL_ACCESSOR_SUBCLASS(accessor_opdata_r, accessor_subscript_w,
                       accessMode == access::mode::read && dimensions == 0) {
  using PureType = typename detail::remove_AS<dataT>::type;
  operator PureType() const {
    return this->__get_impl()->Data[0];
  }
};

SYCL_ACCESSOR_SUBCLASS(accessor_subscript_rn, accessor_opdata_r,
                       accessMode == access::mode::read && dimensions > 0) {
  typename detail::remove_AS<dataT>::type
  operator[](id<dimensions> index) const {
    return this->__get_impl()->Data[getOffsetForId(
      this->__get_impl()->MemRange, index, this->get_offset())];
  }

  subscript_obj<dimensions, dataT, dimensions - 1, accessMode, accessTarget,
              isPlaceholder>
  operator[](size_t index) const {
    id<dimensions> ids;
    ids[0] = index;
    return subscript_obj<dimensions, dataT, dimensions - 1, accessMode,
                         accessTarget, isPlaceholder>(*this, ids);
  }
};

SYCL_ACCESSOR_SUBCLASS(accessor_subscript_r, accessor_subscript_rn,
                       accessMode == access::mode::read && dimensions == 1) {
  typename detail::remove_AS<dataT>::type
  operator[](id<dimensions> index) const {
    return this->operator[](
      getOffsetForId(this->__get_impl()->MemRange, index, this->get_offset()));
  }
  typename detail::remove_AS<dataT>::type
  operator[](size_t index) const {
    return this->__get_impl()->Data[index];
  }
};

template <access::target accessTarget> struct getAddressSpace {
  constexpr static cl::sycl::access::address_space value =
      cl::sycl::access::address_space::global_space;
};

template <> struct getAddressSpace<access::target::local> {
  constexpr static cl::sycl::access::address_space value =
      cl::sycl::access::address_space::local_space;
};

// Available when: accessMode == access::mode::atomic && dimensions == 0
SYCL_ACCESSOR_SUBCLASS(accessor_subscript_atomic_eq0, accessor_subscript_r,
                       accessMode == access::mode::atomic && dimensions == 0) {
  using PureType = typename detail::remove_AS<dataT>::type;
  constexpr static access::address_space addressSpace =
      getAddressSpace<accessTarget>::value;
  operator atomic<PureType, addressSpace>() const {
    return atomic<PureType, addressSpace>(
        multi_ptr<PureType, addressSpace>(&(this->__get_impl()->Data[0])));
  }
};

// Available when: accessMode == access::mode::atomic && dimensions > 0
SYCL_ACCESSOR_SUBCLASS(accessor_subscript_atomic_gt0,
                       accessor_subscript_atomic_eq0,
                       accessMode == access::mode::atomic && dimensions > 0) {
  using PureType = typename detail::remove_AS<dataT>::type;
  constexpr static access::address_space addressSpace =
      getAddressSpace<accessTarget>::value;
  atomic<PureType, addressSpace> operator[](id<dimensions> index) const {
    return atomic<PureType, addressSpace>(
        multi_ptr<PureType, addressSpace>(&(this->__get_impl()->Data[getOffsetForId(
            this->__get_impl()->MemRange, index, this->__get_impl()->Offset)])));
  }
};

// Available only when: accessMode == access::mode::atomic && dimensions == 1
SYCL_ACCESSOR_SUBCLASS(accessor_subscript_atomic_eq1,
                       accessor_subscript_atomic_gt0,
                       accessMode == access::mode::atomic && dimensions == 1) {
  using PureType = typename detail::remove_AS<dataT>::type;
  constexpr static access::address_space addressSpace =
      getAddressSpace<accessTarget>::value;
  atomic<PureType, addressSpace> operator[](size_t index) const {
    return atomic<PureType, addressSpace>(
        multi_ptr<PureType, addressSpace>(&(this->__get_impl()->Data[index])));
  }
};

// TODO:
// /* Available only when: dimensions > 1 */
// __unspecified__ &operator[](size_t index) const;

SYCL_ACCESSOR_SUBCLASS(accessor_pointer, accessor_subscript_atomic_eq1, true) {
  /* Available only when: accessTarget == access::target::host_buffer */
  template <typename DataT = typename detail::remove_AS<dataT>::type,
            access::target AccessTarget = accessTarget>
  typename std::enable_if<(AccessTarget == access::target::host_buffer),
                          dataT *>::type
  get_pointer() const {
    return this->__get_impl()->Data;
  }
  /* Available only when: accessTarget == access::target::global_buffer */
  template <typename DataT = typename detail::remove_AS<dataT>::type,
            access::target AccessTarget = accessTarget>
  typename std::enable_if<(AccessTarget == access::target::global_buffer),
                          global_ptr<DataT>>::type
  get_pointer() const {
    return global_ptr<DataT>(this->__get_impl()->Data);
  }
  /* Available only when: accessTarget == access::target::constant_buffer */
  template <typename DataT = typename detail::remove_AS<dataT>::type,
            access::target AccessTarget = accessTarget>
  typename std::enable_if<(AccessTarget == access::target::constant_buffer),
                          constant_ptr<DataT>>::type
  get_pointer() const {
    return constant_ptr<DataT>(this->__get_impl()->Data);
  }
  /* Available only when: accessTarget == access::target::local */
  template <typename DataT = typename detail::remove_AS<dataT>::type,
            access::target AccessTarget = accessTarget>
  typename std::enable_if<(AccessTarget == access::target::local),
                          local_ptr<DataT>>::type
  get_pointer() const {
    return local_ptr<DataT>(this->__get_impl()->Data);
  }
};

} // namespace detail

//
// Actual definition of sycl::accessor class.
//
template <typename dataT, int dimensions, access::mode accessMode,
          access::target accessTarget = access::target::global_buffer,
          access::placeholder isPlaceholder = access::placeholder::false_t>
class accessor
    : public detail::accessor_pointer<
          typename detail::DeviceValueType<dataT, accessTarget>::type,
          dimensions, accessMode, accessTarget, isPlaceholder> {
  using _ValueType =
      typename detail::DeviceValueType<dataT, accessTarget>::type;
  using _ImplT = detail::accessor_impl<_ValueType, dimensions, accessMode,
                                       accessTarget, isPlaceholder>;

  // Make sure Impl field is the first in the class, so that it is
  // safe to reinterpret a pointer to accessor as a pointer to the
  // implementation.
#ifdef __SYCL_DEVICE_ONLY__
  _ImplT impl;
#else
  std::shared_ptr<_ImplT> impl;
  char padding[sizeof(_ImplT) - sizeof(std::shared_ptr<_ImplT>)];
#endif

#ifdef __SYCL_DEVICE_ONLY__
  void __init(_ValueType *Ptr, range<dimensions> AccessRange,
              range<dimensions> MemRange, id<dimensions> Offset) {
    impl.Data = Ptr;
    impl.AccessRange = AccessRange;
    impl.MemRange = MemRange;
    impl.Offset = Offset;
  }
#endif

#ifndef __SYCL_DEVICE_ONLY__
  detail::buffer_impl<buffer_allocator> *getBufImpl() const {
    return impl->m_Buf;
  }
#endif

  range<dimensions> getAccessRange() const {
#ifdef __SYCL_DEVICE_ONLY__
    return impl.AccessRange;
#else
    return impl->AccessRange;
#endif
  };

  range<dimensions> getMemRange() const {
#ifdef __SYCL_DEVICE_ONLY__
    return impl.MemRange;
#else
    return impl->MemRange;
#endif
  };

  id<dimensions> getOffset() const {
#ifdef __SYCL_DEVICE_ONLY__
    return impl.Offset;
#else
    return impl->Offset;
#endif
  };

  size_t getByteSize() const {
#ifdef __SYCL_DEVICE_ONLY__
    return impl.ByteSize;
#else
    return impl->ByteSize;
#endif
  };

  template <typename KernelType, int Dimensions, typename RangeType,
            typename KernelArgType, bool SingleTask>
  friend class cl::sycl::simple_scheduler::ExecuteKernelCommand;

  template <int AccessDimensions, typename KernelType>
  friend uint cl::sycl::simple_scheduler::passGlobalAccessorAsArg(
      uint I, int LambdaOffset, cl_kernel ClKernel,
      const KernelType &HostKernel);

  template <int AccessDimensions, typename KernelType>
  friend uint cl::sycl::simple_scheduler::passLocalAccessorAsArg(
      uint I, int LambdaOffset, cl_kernel ClKernel,
      const KernelType &HostKernel);

  template <class Obj>
  friend decltype(Obj::impl) detail::getSyclObjImpl(const Obj &SyclObject);

public:
  using value_type = dataT;
  using reference = dataT &;
  using const_reference = const dataT &;

  // buffer accessor ctor #1
  //   accessor(buffer<dataT, 1> &);
  //
  // Available only when:
  //   ((isPlaceholder == access::placeholder::false_t &&
  //     accessTarget == access::target::host_buffer) ||
  //    (isPlaceholder == access::placeholder::true_t  &&
  //     (accessTarget == access::target::global_buffer||
  //      accessTarget == access::target::constant_buffer))) &&
  //   dimensions == 0
  template <typename DataT = dataT, int Dimensions = dimensions,
            access::mode AccessMode = accessMode,
            access::target AccessTarget = accessTarget,
            access::placeholder IsPlaceholder = isPlaceholder>
  accessor(typename std::enable_if<
           (((IsPlaceholder == access::placeholder::false_t &&
              AccessTarget == access::target::host_buffer) ||
             (IsPlaceholder == access::placeholder::true_t  &&
               (AccessTarget == access::target::global_buffer ||
                AccessTarget == access::target::constant_buffer))) &&
            Dimensions == 0),
           buffer<DataT, 1>>::type &bufferRef)
#ifdef __SYCL_DEVICE_ONLY__
      : impl((dataT *)detail::getSyclObjImpl(bufferRef)->BufPtr) {
#else
      : impl(std::make_shared<_ImplT>(
                 (dataT *)detail::getSyclObjImpl(bufferRef)->BufPtr)) {
#endif
    auto BufImpl = detail::getSyclObjImpl(bufferRef);
    if (AccessTarget == access::target::host_buffer) {
      if (BufImpl->OpenCLInterop) {
        throw cl::sycl::runtime_error(
            "Host access to interoperability buffer is not allowed");
      } else {
        simple_scheduler::Scheduler::getInstance()
            .copyBack<AccessMode, AccessTarget>(*BufImpl);
      }
    }
    if (BufImpl->OpenCLInterop && !BufImpl->isValidAccessToMem(accessMode)) {
      throw cl::sycl::runtime_error(
          "Access mode is incompatible with opencl memory object of the "
          "interoperability buffer");
    }
  }

  // buffer accessor ctor #2:
  //   accessor(buffer<dataT, 1> &, handler &);
  //
  // Available only when:
  //   isPlaceholder == access::placeholder::false_t &&
  //   (accessTarget == access::target::global_buffer ||
  //    accessTarget == access::target::constant_buffer) &&
  //   dimensions == 0
  template <typename DataT = dataT, int Dimensions = dimensions,
            access::mode AccessMode = accessMode,
            access::target AccessTarget = accessTarget,
            access::placeholder IsPlaceholder = isPlaceholder>
  accessor(typename std::enable_if<
           (IsPlaceholder == access::placeholder::false_t &&
            (AccessTarget == access::target::global_buffer ||
             AccessTarget == access::target::constant_buffer) &&
             Dimensions == 0),
           buffer<DataT, 1>>::type &bufferRef,
           handler &commandGroupHandlerRef)
#ifdef __SYCL_DEVICE_ONLY__
      // Even though this ctor can not be used in device code, some
      // dummy implementation is still needed.
      // Pass nullptr as a pointer to mem and use buffers from the ctor
      // arguments to avoid the need in adding utility functions for
      // dummy/default initialization of range fields.
      : impl(nullptr, (handler *)nullptr) {}
#else // !__SYCL_DEVICE_ONLY__
      : impl(std::make_shared<_ImplT>(
                 (dataT *)detail::getSyclObjImpl(bufferRef)->BufPtr,
                 &commandGroupHandlerRef)) {
    auto BufImpl = detail::getSyclObjImpl(bufferRef);
    if (BufImpl->OpenCLInterop && !BufImpl->isValidAccessToMem(accessMode)) {
      throw cl::sycl::runtime_error(
          "Access mode is incompatible with opencl memory object of the "
          "interoperability buffer");
    }
    commandGroupHandlerRef.AddBufDep<AccessMode, AccessTarget>(*BufImpl);
    impl->m_Buf = BufImpl.get();
  }
#endif // !__SYCL_DEVICE_ONLY__

  // buffer accessor ctor #3:
  //   accessor(buffer &);
  //
  // Available only when:
  //   ((isPlaceholder == access::placeholder::false_t &&
  //     accessTarget == access::target::host_buffer) ||
  //    (isPlaceholder == access::placeholder::true_t &&
  //     (accessTarget == access::target::global_buffer ||
  //      accessTarget == access::target::constant_buffer))) &&
  //   dimensions > 0)
  template <typename DataT = dataT, int Dimensions = dimensions,
            access::mode AccessMode = accessMode,
            access::target AccessTarget = accessTarget,
            access::placeholder IsPlaceholder = isPlaceholder>
  accessor(typename std::enable_if<
           (((IsPlaceholder == access::placeholder::false_t &&
              AccessTarget == access::target::host_buffer) ||
             (IsPlaceholder == access::placeholder::true_t &&
              (AccessTarget == access::target::global_buffer ||
               AccessTarget == access::target::constant_buffer))) &&
            Dimensions > 0),
           buffer<DataT, Dimensions>>::type &bufferRef)
#ifdef __SYCL_DEVICE_ONLY__
      : impl((dataT *)detail::getSyclObjImpl(bufferRef)->BufPtr,
             bufferRef.MemRange, bufferRef.MemRange) {
#else
      : impl(std::make_shared<_ImplT>(
                 (dataT *)detail::getSyclObjImpl(bufferRef)->BufPtr,
                 bufferRef.MemRange, bufferRef.MemRange)) {
#endif
    auto BufImpl = detail::getSyclObjImpl(bufferRef);
    if (AccessTarget == access::target::host_buffer) {
      if (BufImpl->OpenCLInterop) {
        throw cl::sycl::runtime_error(
            "Host access to interoperability buffer is not allowed");
      } else {
        simple_scheduler::Scheduler::getInstance()
            .copyBack<AccessMode, AccessTarget>(*BufImpl);
      }
    }
    if (BufImpl->OpenCLInterop && !BufImpl->isValidAccessToMem(accessMode)) {
      throw cl::sycl::runtime_error(
          "Access mode is incompatible with opencl memory object of the "
          "interoperability buffer");
    }
  }

  // buffer ctor #4:
  //   accessor(buffer &, handler &);
  //
  // Available only when:
  //   isPlaceholder == access::placeholder::false_t &&
  //   (accessTarget == access::target::global_buffer ||
  //    accessTarget == access::target::constant_buffer) &&
  //   dimensions > 0
  template <typename DataT = dataT, int Dimensions = dimensions,
            access::mode AccessMode = accessMode,
            access::target AccessTarget = accessTarget,
            access::placeholder IsPlaceholder = isPlaceholder>
  accessor(typename std::enable_if<
               (IsPlaceholder == access::placeholder::false_t &&
                (AccessTarget == access::target::global_buffer ||
                 AccessTarget == access::target::constant_buffer) &&
                Dimensions > 0),
               buffer<DataT, Dimensions>>::type &bufferRef,
           handler &commandGroupHandlerRef)
#ifdef __SYCL_DEVICE_ONLY__
      // Even though this ctor can not be used in device code, some
      // dummy implementation is still needed.
      // Pass nullptr as a pointer to mem and use buffers from the ctor
      // arguments to avoid the need in adding utility functions for
      // dummy/default initialization of range fields.
      : impl(nullptr, bufferRef.MemRange, bufferRef.MemRange,
             &commandGroupHandlerRef) {}
#else
      : impl(std::make_shared<_ImplT>(
                 (dataT *)detail::getSyclObjImpl(bufferRef)->BufPtr,
                 bufferRef.MemRange, bufferRef.MemRange,
                 &commandGroupHandlerRef)) {
    auto BufImpl = detail::getSyclObjImpl(bufferRef);
    if (BufImpl->OpenCLInterop && !BufImpl->isValidAccessToMem(accessMode)) {
      throw cl::sycl::runtime_error(
          "Access mode is incompatible with opencl memory object of the "
          "interoperability buffer");
    }
    commandGroupHandlerRef.AddBufDep<AccessMode, AccessTarget>(*BufImpl);
    impl->m_Buf = BufImpl.get();
  }
#endif

  // accessor ctor #5:
  //   accessor(buffer &, range Range, id Offset = {});
  //
  // Available only when:
  //   (isPlaceholder == access::placeholder::false_t &&
  //    accessTarget == access::target::host_buffer) ||
  //   (isPlaceholder == access::placeholder::true_t &&
  //    (accessTarget == access::target::global_buffer ||
  //     accessTarget == access::target::constant_buffer) &&
  //    dimensions > 0)
  template <typename DataT = dataT, int Dimensions = dimensions,
            access::mode AccessMode = accessMode,
            access::target AccessTarget = accessTarget,
            access::placeholder IsPlaceholder = isPlaceholder>
  accessor(typename std::enable_if<
               ((IsPlaceholder == access::placeholder::false_t &&
                 AccessTarget == access::target::host_buffer) ||
                (IsPlaceholder == access::placeholder::true_t &&
                 (AccessTarget == access::target::global_buffer ||
                  AccessTarget == access::target::constant_buffer) &&
                 Dimensions > 0)),
               buffer<DataT, Dimensions>>::type &bufferRef,
           range<Dimensions> Range, id<Dimensions> Offset = {})
#ifdef __SYCL_DEVICE_ONLY__
      // Even though this ctor can not be used in device code, some
      // dummy implementation is still needed.
      // Pass nullptr as a pointer to mem and use buffers from the ctor
      // arguments to avoid the need in adding utility functions for
      // dummy/default initialization of range<Dimensions> and
      // id<Dimension> fields.
      : impl(nullptr, Range, bufferRef.MemRange, Offset) {}
#else   // !__SYCL_DEVICE_ONLY__
      : impl(std::make_shared<_ImplT>(
                 (dataT *)detail::getSyclObjImpl(bufferRef)->BufPtr, Range,
                 bufferRef.MemRange, Offset)) {
    auto BufImpl = detail::getSyclObjImpl(bufferRef);
    if (AccessTarget == access::target::host_buffer) {
      if (BufImpl->OpenCLInterop) {
        throw cl::sycl::runtime_error(
            "Host access to interoperability buffer is not allowed");
      } else {
        simple_scheduler::Scheduler::getInstance()
            .copyBack<AccessMode, AccessTarget>(*BufImpl);
      }
    }
    if (BufImpl->OpenCLInterop && !BufImpl->isValidAccessToMem(accessMode)) {
      throw cl::sycl::runtime_error(
          "Access mode is incompatible with opencl memory object of the "
          "interoperability buffer");
    }
  }
#endif // !__SYCL_DEVICE_ONLY__

  // buffer ctor #6:
  //   accessor(buffer &, handler &, range Range, id Offset);
  //
  // Available only when:
  //   isPlaceholder == access::placeholder::false_t &&
  //   (accessTarget == access::target::global_buffer ||
  //    accessTarget == access::target::constant_buffer) &&
  //   dimensions > 0
  template <typename DataT = dataT, int Dimensions = dimensions,
            access::mode AccessMode = accessMode,
            access::target AccessTarget = accessTarget,
            access::placeholder IsPlaceholder = isPlaceholder>
  accessor(typename std::enable_if<
               (IsPlaceholder == access::placeholder::false_t &&
                (AccessTarget == access::target::global_buffer ||
                 AccessTarget == access::target::constant_buffer) &&
                Dimensions > 0),
               buffer<DataT, Dimensions>>::type &bufferRef,
           handler &commandGroupHandlerRef, range<Dimensions> Range,
           id<Dimensions> Offset = {})
#ifdef __SYCL_DEVICE_ONLY__
      // Even though this ctor can not be used in device code, some
      // dummy implementation is still needed.
      // Pass nullptr as a pointer to mem and use buffers from the ctor
      // arguments to avoid the need in adding utility functions for
      // dummy/default initialization of range<Dimensions> and
      // id<Dimension> fields.
      : impl(nullptr, Range, bufferRef.MemRange,
             &commandGroupHandlerRef, Offset) {}
#else   // !__SYCL_DEVICE_ONLY__
      : impl(std::make_shared<_ImplT>(
                 (dataT *)detail::getSyclObjImpl(bufferRef)->BufPtr, Range,
                 bufferRef.MemRange, &commandGroupHandlerRef, Offset)) {
    auto BufImpl = detail::getSyclObjImpl(bufferRef);
    if (BufImpl->OpenCLInterop && !BufImpl->isValidAccessToMem(accessMode)) {
      throw cl::sycl::runtime_error(
          "Access mode is incompatible with opencl memory object of the "
          "interoperability buffer");
    }
    commandGroupHandlerRef.AddBufDep<AccessMode, AccessTarget>(*BufImpl);
    impl->m_Buf = BufImpl.get();
  }
#endif // !__SYCL_DEVICE_ONLY__

  // TODO:
  // local accessor ctor #1
  // accessor(handler &);
  // Available only when:
  //   AccessTarget == access::target::local && Dimensions == 0
  //
  // template <typename DataT = dataT, int Dimensions = dimensions,
  //           access::mode AccessMode = accessMode,
  //           access::target AccessTarget = accessTarget,
  //           access::placeholder IsPlaceholder = isPlaceholder>
  // accessor(typename std::enable_if<(AccessTarget == access::target::local &&
  // Dimensions == 0), handler>::type &commandGroupHandlerRef);


  // local accessor ctor #2
  // accessor(range allocationSize, handler &);
  // Available only when:
  //   AccessTarget == access::target::local && Dimensions => 0
  template <typename DataT = dataT, int Dimensions = dimensions,
            access::mode AccessMode = accessMode,
            access::target AccessTarget = accessTarget,
            access::placeholder IsPlaceholder = isPlaceholder>
  accessor(typename std::enable_if<(AccessTarget == access::target::local &&
                                    Dimensions > 0),
                                   range<Dimensions>>::type allocationSize,
           handler &commandGroupHandlerRef)
#ifdef __SYCL_DEVICE_ONLY__
      : impl(allocationSize, &commandGroupHandlerRef) {}
#else
      : impl(std::make_shared<_ImplT>(allocationSize,
                                      &commandGroupHandlerRef)) {}
#endif

  accessor(const accessor &rhs) = default;
  accessor(accessor &&rhs) = default;

  accessor &operator=(const accessor &rhs) = default;
  accessor &operator=(accessor &&rhs) = default;

  ~accessor() = default;

  bool operator==(const accessor &rhs) const { return impl == rhs.impl; }
  bool operator!=(const accessor &rhs) const { return !(*this == rhs); }
};

} // namespace sycl
} // namespace cl

#undef SYCL_ACCESSOR_IMPL
#undef SYCL_ACCESSOR_SUBCLASS

namespace std {
template <typename T, int Dimensions, cl::sycl::access::mode AccessMode,
          cl::sycl::access::target AccessTarget,
          cl::sycl::access::placeholder IsPlaceholder>
struct hash<cl::sycl::accessor<T, Dimensions, AccessMode, AccessTarget,
                               IsPlaceholder>> {
  using AccType = cl::sycl::accessor<
      T, Dimensions, AccessMode, AccessTarget, IsPlaceholder>;
  using ImplType = cl::sycl::detail::accessor_impl<
      T, Dimensions, AccessMode, AccessTarget, IsPlaceholder>;

  size_t operator()(const AccType &A) const {
#ifdef __SYCL_DEVICE_ONLY__
    // Hash is not supported on DEVICE. Just return 0 here.
    return 0;
#else
    std::shared_ptr<ImplType> ImplPtr = cl::sycl::detail::getSyclObjImpl(A);
    return hash<std::shared_ptr<ImplType>>()(ImplPtr);
#endif
  }
};
} // namespace std
