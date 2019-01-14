//==--------- accessor.hpp --- SYCL accessor -------------------------------==//
//
// The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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

namespace cl {
namespace sycl {
// TODO: 4.3.2 Implement common reference semantics
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

  INLINE_IF_DEVICE subscript_obj<accessorDim, dataT, dimensions - 1, accessMode, accessTarget,
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

  INLINE_IF_DEVICE dataT &operator[](size_t index) {
    ids[accessorDim - 1] = index;
    return accRef.__impl()->Data[getOffsetForId(
      accRef.__impl()->Range, ids, accRef.__impl()->Offset)];
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

  INLINE_IF_DEVICE typename detail::remove_AS<dataT>::type
  operator[](size_t index) {
    ids[accessorDim - 1] = index;
    return accRef.__impl()->Data[getOffsetForId(
      accRef.__impl()->Range, ids, accRef.__impl()->Offset)];
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
  accessor_impl(dataT *Data) : Data(Data) {}

  // Returns the number of accessed elements.
  INLINE_IF_DEVICE size_t get_count() const { return 1; }
};

/// Implementation of host accessor.
/// Available when (dimensions > 0).
SYCL_ACCESSOR_IMPL(isTargetHostAccess(accessTarget) && dimensions > 0) {
  dataT *Data;
  range<dimensions> Range;
  id<dimensions> Offset;

  accessor_impl(dataT *Data, range<dimensions> Range,
                id<dimensions> Offset = {})
      : Data(Data), Range(Range), Offset(Offset) {}

  // Returns the number of accessed elements.
  INLINE_IF_DEVICE size_t get_count() const { return Range.size(); }
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
  detail::buffer_impl<dataT, 1> *m_Buf = nullptr;

#else
  char padding[sizeof(detail::buffer_impl<dataT, dimensions> *)];
#endif // __SYCL_DEVICE_ONLY__

  dataT *Data;

  // Device accessors must be associated with a command group handler.
  // The handler though can be nullptr at the creation point if the
  // accessor is a placeholder accessor.
  accessor_impl(dataT *Data, handler *Handler = nullptr)
      : Data(Data)
  {}

  // Returns the number of accessed elements.
  INLINE_IF_DEVICE size_t get_count() const { return 1; }

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
  detail::buffer_impl<dataT, dimensions> *m_Buf = nullptr;
#else
  char padding[sizeof(detail::buffer_impl<dataT, dimensions> *)];
#endif // __SYCL_DEVICE_ONLY__

  dataT *Data;
  range<dimensions> Range;
  id<dimensions> Offset;

  // Device accessors must be associated with a command group handler.
  // The handler though can be nullptr at the creation point if the
  // accessor is a placeholder accessor.
  accessor_impl(dataT *Data, range<dimensions> Range,
                handler *Handler = nullptr, id<dimensions> Offset = {})
      : Data(Data), Range(Range), Offset(Offset)
  {}

  // Returns the number of accessed elements.
  INLINE_IF_DEVICE size_t get_count() const { return Range.size(); }

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
  INLINE_IF_DEVICE size_t get_count() const { return 1; }

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
  range<dimensions> Range;
  // TODO delete it when accessor class was remade
  // Offset field is not need for local accessor, but this field is now used
  // in the inheritance hierarchy. Getting rid of this field will cause
  // duplication and complication of the code even more.
  id<dimensions> Offset;

  accessor_impl(range<dimensions> Range, handler * Handler) : Range(Range),
      ByteSize(Range.size() * sizeof(dataT))
  {
#ifndef __SYCL_DEVICE_ONLY__
    assert(Handler != nullptr && "Handler is nullptr");
    if (Handler->is_host()) {
      dataBuf = std::make_shared<vector_class<dataT>>(Range.size());
      Data = dataBuf->data();
    }
#endif
  }

  // Returns the number of accessed elements.
  INLINE_IF_DEVICE size_t get_count() const { return Range.size(); }

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

  INLINE_IF_DEVICE const _ImplT *__impl() const {
    return reinterpret_cast<const _ImplT *>(this);
  }

  INLINE_IF_DEVICE _ImplT *__impl() { return reinterpret_cast<_ImplT *>(this); }

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
  INLINE_IF_DEVICE constexpr bool is_placeholder() const {
    return isPlaceholder == access::placeholder::true_t;
  }

  // Returns the size of the accessed memory in bytes.
  INLINE_IF_DEVICE size_t get_size() const { return this->get_count() * sizeof(dataT); }

  // Returns the number of accessed elements.
  INLINE_IF_DEVICE size_t get_count() const { return this->__impl()->get_count(); }

  template <int Dimensions = dimensions> INLINE_IF_DEVICE
  typename std::enable_if<(Dimensions > 0), range<Dimensions>>::type
  get_range() const { return this->__impl()->Range; }

  template <int Dimensions = dimensions> INLINE_IF_DEVICE
  typename std::enable_if<(Dimensions > 0), id<Dimensions>>::type
  get_offset() const { return this->__impl()->Offset; }
};

SYCL_ACCESSOR_SUBCLASS(accessor_opdata_w, accessor_common,
                       (accessMode == access::mode::write ||
                        accessMode == access::mode::read_write ||
                        accessMode == access::mode::discard_write ||
                        accessMode == access::mode::discard_read_write) &&
                       dimensions == 0) {
  INLINE_IF_DEVICE operator dataT &() const {
    return this->__impl()->Data[0];
  }
};

SYCL_ACCESSOR_SUBCLASS(accessor_subscript_wn, accessor_opdata_w,
                       (accessMode == access::mode::write ||
                        accessMode == access::mode::read_write ||
                        accessMode == access::mode::discard_write ||
                        accessMode == access::mode::discard_read_write) &&
                       dimensions > 0) {
  dataT &operator[](id<dimensions> index) const {
    return this->__impl()->Data[getOffsetForId(
      this->get_range(), index, this->get_offset())];
  }

  subscript_obj<dimensions, dataT, dimensions - 1, accessMode, accessTarget,
              isPlaceholder>
  INLINE_IF_DEVICE operator[](size_t index) const {
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
  INLINE_IF_DEVICE dataT &operator[](id<dimensions> index) const {
    return this->operator[](
      getOffsetForId(this->get_range(), index, this->get_offset()));
  }
  INLINE_IF_DEVICE dataT &operator[](size_t index) const {
    return this->__impl()->Data[index];
  }
};

SYCL_ACCESSOR_SUBCLASS(accessor_opdata_r, accessor_subscript_w,
                       accessMode == access::mode::read && dimensions == 0) {
  using PureType = typename detail::remove_AS<dataT>::type;
  operator PureType() const {
    return this->__impl()->Data[0];
  }
};

SYCL_ACCESSOR_SUBCLASS(accessor_subscript_rn, accessor_opdata_r,
                       accessMode == access::mode::read && dimensions > 0) {
  typename detail::remove_AS<dataT>::type
  operator[](id<dimensions> index) const {
    return this->__impl()->Data[getOffsetForId(
      this->get_range(), index, this->get_offset())];
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
      getOffsetForId(this->get_range(), index, this->get_offset()));
  }
  typename detail::remove_AS<dataT>::type
  operator[](size_t index) const {
    return this->__impl()->Data[index];
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
        multi_ptr<PureType, addressSpace>(&(this->__impl()->Data[0])));
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
        multi_ptr<PureType, addressSpace>(&(this->__impl()->Data[getOffsetForId(
            this->__impl()->Range, index, this->__impl()->Offset)])));
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
        multi_ptr<PureType, addressSpace>(&(this->__impl()->Data[index])));
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
    return this->__impl()->Data;
  }
  /* Available only when: accessTarget == access::target::global_buffer */
  template <typename DataT = typename detail::remove_AS<dataT>::type,
            access::target AccessTarget = accessTarget>
  typename std::enable_if<(AccessTarget == access::target::global_buffer),
                          global_ptr<DataT>>::type
  get_pointer() const {
    return global_ptr<DataT>(this->__impl()->Data);
  }
  /* Available only when: accessTarget == access::target::constant_buffer */
  template <typename DataT = typename detail::remove_AS<dataT>::type,
            access::target AccessTarget = accessTarget>
  typename std::enable_if<(AccessTarget == access::target::constant_buffer),
                          constant_ptr<DataT>>::type
  get_pointer() const {
    return constant_ptr<DataT>(this->__impl()->Data);
  }
  /* Available only when: accessTarget == access::target::local */
  template <typename DataT = typename detail::remove_AS<dataT>::type,
            access::target AccessTarget = accessTarget>
  typename std::enable_if<(AccessTarget == access::target::local),
                          local_ptr<DataT>>::type
  get_pointer() const {
    return local_ptr<DataT>(this->__impl()->Data);
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
  _ImplT __impl;

  INLINE_IF_DEVICE void __init(_ValueType *Ptr, range<dimensions> Range,
      id<dimensions> Offset) {
    __impl.Data = Ptr;
    __impl.Range = Range;
    __impl.Offset = Offset;
  }

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
      : __impl(detail::getSyclObjImpl(bufferRef)->BufPtr) {
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
      ; // This ctor can't be used in device code, so no need to define it.
#else // !__SYCL_DEVICE_ONLY__
      : __impl(detail::getSyclObjImpl(bufferRef)->BufPtr,
               detail::getSyclObjImpl(bufferRef)->Range,
               &commandGroupHandlerRef) {
    auto BufImpl = detail::getSyclObjImpl(bufferRef);
    if (BufImpl->OpenCLInterop && !BufImpl->isValidAccessToMem(accessMode)) {
      throw cl::sycl::runtime_error(
          "Access mode is incompatible with opencl memory object of the "
          "interoperability buffer");
    }
    commandGroupHandlerRef.AddBufDep<AccessMode, AccessTarget>(*BufImpl);
    __impl.m_Buf = BufImpl.get();
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
      : __impl(detail::getSyclObjImpl(bufferRef)->BufPtr,
               detail::getSyclObjImpl(bufferRef)->Range) {
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
      ; // This ctor can't be used in device code, so no need to define it.
#else
      : __impl(detail::getSyclObjImpl(bufferRef)->BufPtr,
               detail::getSyclObjImpl(bufferRef)->Range,
               &commandGroupHandlerRef) {
    auto BufImpl = detail::getSyclObjImpl(bufferRef);
    if (BufImpl->OpenCLInterop && !BufImpl->isValidAccessToMem(accessMode)) {
      throw cl::sycl::runtime_error(
          "Access mode is incompatible with opencl memory object of the "
          "interoperability buffer");
    }
    commandGroupHandlerRef.AddBufDep<AccessMode, AccessTarget>(*BufImpl);
    __impl.m_Buf = BufImpl.get();
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
           range<Dimensions> Range,
           id<Dimensions> Offset = {}
          )
#ifdef __SYCL_DEVICE_ONLY__
      ; // This ctor can't be used in device code, so no need to define it.
#else // !__SYCL_DEVICE_ONLY__
      : __impl(detail::getSyclObjImpl(bufferRef)->BufPtr, Range, Offset) {
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
           handler &commandGroupHandlerRef,
           range<Dimensions> Range,
           id<Dimensions> Offset = {}
          )
#ifdef __SYCL_DEVICE_ONLY__
      ; // This ctor can't be used in device code, so no need to define it.
#else // !__SYCL_DEVICE_ONLY__
      : __impl(detail::getSyclObjImpl(bufferRef)->BufPtr, Range,
               &commandGroupHandlerRef, Offset) {
    auto BufImpl = detail::getSyclObjImpl(bufferRef);
    if (BufImpl->OpenCLInterop && !BufImpl->isValidAccessToMem(accessMode)) {
      throw cl::sycl::runtime_error(
          "Access mode is incompatible with opencl memory object of the "
          "interoperability buffer");
    }
    commandGroupHandlerRef.AddBufDep<AccessMode, AccessTarget>(*BufImpl);
    __impl.m_Buf = BufImpl.get();
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
      : __impl(allocationSize, &commandGroupHandlerRef) {}
};

} // namespace sycl
} // namespace cl

#undef SYCL_ACCESSOR_IMPL
#undef SYCL_ACCESSOR_SUBCLASS

//TODO hash for accessor
