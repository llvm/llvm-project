//==---------------- atomic.hpp - SYCL atomics -----------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once
#include <CL/sycl/access/access.hpp>
#ifdef __SYCL_DEVICE_ONLY__
#include <CL/__spirv/spirv_ops.hpp>
#else
#include <CL/__spirv/spirv_types.hpp>
#include <atomic>
#endif
#include <type_traits>

#define STATIC_ASSERT_NOT_FLOAT(T)                                             \
  static_assert(!std::is_same<T, float>::value,                                \
                "SYCL atomic function not available for float type")

namespace cl {
namespace sycl {

enum class memory_order : int { relaxed };

// Forward declaration
template <typename pointerT, access::address_space addressSpace>
class multi_ptr;

namespace detail {

using memory_order = cl::sycl::memory_order;

template <typename T> struct IsValidAtomicType {
  static constexpr bool value =
      (std::is_same<T, int>::value || std::is_same<T, unsigned int>::value ||
       std::is_same<T, long>::value || std::is_same<T, unsigned long>::value ||
       std::is_same<T, long long>::value ||
       std::is_same<T, unsigned long long>::value ||
       std::is_same<T, float>::value);
};

template <cl::sycl::access::address_space AS> struct IsValidAtomicAddressSpace {
  static constexpr bool value = (AS == access::address_space::global_space ||
                                 AS == access::address_space::local_space);
};

// Type trait to translate a cl::sycl::access::address_space to
// a SPIR-V memory scope
template <access::address_space AS> struct GetSpirvMemoryScope {};
template <> struct GetSpirvMemoryScope<access::address_space::global_space> {
  static constexpr auto scope = cl::__spirv::Scope::Device;
};
template <> struct GetSpirvMemoryScope<access::address_space::local_space> {
  static constexpr auto scope = ::cl::__spirv::Scope::Workgroup;
};

// Translate the cl::sycl::memory_order to a SPIR-V builtin order
static inline ::cl::__spirv::MemorySemantics
getSpirvMemorySemantics(memory_order Order) {
  return ::cl::__spirv::MemorySemantics::None;
}

} // namespace detail
} // namespace sycl
} // namespace cl

#ifndef __SYCL_DEVICE_ONLY__
// host implementation of SYCL atomics
namespace cl {
namespace sycl {
namespace detail {
// Translate cl::sycl::memory_order or cl::__spirv::MemorySemantics
// into std::memory_order
// Only relaxed memory semantics are supported currently
static inline std::memory_order
getStdMemoryOrder(::cl::__spirv::MemorySemantics MS) {
  return std::memory_order_relaxed;
}
static inline std::memory_order getStdMemoryOrder(::cl::sycl::memory_order MS) {
  return std::memory_order_relaxed;
}
} // namespace detail
} // namespace sycl

// std::atomic version of atomic SPIR-V builtins
namespace __spirv {

template <typename T>
void OpAtomicStore(std::atomic<T> *Ptr, Scope S, MemorySemantics MS, T V) {
  Ptr->store(V, ::cl::sycl::detail::getStdMemoryOrder(MS));
}

template <typename T>
T OpAtomicLoad(const std::atomic<T> *Ptr, Scope S, MemorySemantics MS) {
  return Ptr->load(::cl::sycl::detail::getStdMemoryOrder(MS));
}

template <typename T>
T OpAtomicExchange(std::atomic<T> *Ptr, Scope S, MemorySemantics MS, T V) {
  return Ptr->exchange(V, ::cl::sycl::detail::getStdMemoryOrder(MS));
}

template <typename T>
extern T OpAtomicIAdd(std::atomic<T> *Ptr, Scope S, MemorySemantics MS, T V) {
  return Ptr->fetch_add(V, ::cl::sycl::detail::getStdMemoryOrder(MS));
}

template <typename T>
extern T OpAtomicISub(std::atomic<T> *Ptr, Scope S, MemorySemantics MS, T V) {
  return Ptr->fetch_sub(V, ::cl::sycl::detail::getStdMemoryOrder(MS));
}

template <typename T>
extern T OpAtomicAnd(std::atomic<T> *Ptr, Scope S, MemorySemantics MS, T V) {
  return Ptr->fetch_and(V, ::cl::sycl::detail::getStdMemoryOrder(MS));
}

template <typename T>
extern T OpAtomicOr(std::atomic<T> *Ptr, Scope S, MemorySemantics MS, T V) {
  return Ptr->fetch_or(V, ::cl::sycl::detail::getStdMemoryOrder(MS));
}

template <typename T>
extern T OpAtomicXor(std::atomic<T> *Ptr, Scope S, MemorySemantics MS, T V) {
  return Ptr->fetch_xor(V, ::cl::sycl::detail::getStdMemoryOrder(MS));
}

template <typename T>
extern T OpAtomicMin(std::atomic<T> *Ptr, Scope S, MemorySemantics MS, T V) {
  std::memory_order MemoryOrder = ::cl::sycl::detail::getStdMemoryOrder(MS);
  T Val = Ptr->load(MemoryOrder);
  while (V < Val) {
    if (Ptr->compare_exchange_strong(Val, V, MemoryOrder, MemoryOrder))
      break;
    Val = Ptr->load(MemoryOrder);
  }
  return Val;
}

template <typename T>
extern T OpAtomicMax(std::atomic<T> *Ptr, Scope S, MemorySemantics MS, T V) {
  std::memory_order MemoryOrder = ::cl::sycl::detail::getStdMemoryOrder(MS);
  T Val = Ptr->load(MemoryOrder);
  while (V > Val) {
    if (Ptr->compare_exchange_strong(Val, V, MemoryOrder, MemoryOrder))
      break;
    Val = Ptr->load(MemoryOrder);
  }
  return Val;
}

} // namespace __spirv
} // namespace cl
#endif // !defined(__SYCL_DEVICE_ONLY__)

namespace cl {
namespace sycl {

template <typename T, access::address_space addressSpace =
                          access::address_space::global_space>
class atomic {
  static_assert(detail::IsValidAtomicType<T>::value,
                "Invalid SYCL atomic type.  Valid types are: int, "
                "unsigned int, long, unsigned long, long long,  unsigned "
                "long long, float");
  static_assert(detail::IsValidAtomicAddressSpace<addressSpace>::value,
                "Invalid SYCL atomic address_space.  Valid address spaces are: "
                "global_space, local_space");
  static constexpr auto SpirvScope =
      detail::GetSpirvMemoryScope<addressSpace>::scope;

public:
  template <typename pointerT>
#ifdef __SYCL_DEVICE_ONLY__
  atomic(multi_ptr<pointerT, addressSpace> ptr)
      : Ptr(ptr.get())
#else
  atomic(multi_ptr<pointerT, addressSpace> ptr)
      : Ptr(reinterpret_cast<std::atomic<T> *>(ptr.get()))
#endif
  {
    static_assert(sizeof(T) == sizeof(pointerT),
                  "T and pointerT must be same size");
  }

  void store(T Operand, memory_order Order = memory_order::relaxed) {
    ::cl::__spirv::OpAtomicStore(
        Ptr, SpirvScope, detail::getSpirvMemorySemantics(Order), Operand);
  }

  T load(memory_order Order = memory_order::relaxed) const {
    return ::cl::__spirv::OpAtomicLoad(Ptr, SpirvScope,
                                       detail::getSpirvMemorySemantics(Order));
  }

  T exchange(T Operand, memory_order Order = memory_order::relaxed) {
    return ::cl::__spirv::OpAtomicExchange(
        Ptr, SpirvScope, detail::getSpirvMemorySemantics(Order), Operand);
  }

  bool
  compare_exchange_strong(T &Expected, T Desired,
                          memory_order SuccessOrder = memory_order::relaxed,
                          memory_order FailOrder = memory_order::relaxed) {
    STATIC_ASSERT_NOT_FLOAT(T);
#ifdef __SYCL_DEVICE_ONLY__
    T Value = ::cl::__spirv::OpAtomicCompareExchange(
        Ptr, SpirvScope, detail::getSpirvMemorySemantics(SuccessOrder),
        detail::getSpirvMemorySemantics(FailOrder), Desired, Expected);
    return (Value == Expected);
#else
    return Ptr->compare_exchange_strong(Expected, Desired,
                                        detail::getStdMemoryOrder(SuccessOrder),
                                        detail::getStdMemoryOrder(FailOrder));
#endif
  }

  T fetch_add(T Operand, memory_order Order = memory_order::relaxed) {
    STATIC_ASSERT_NOT_FLOAT(T);
    return ::cl::__spirv::OpAtomicIAdd(
        Ptr, SpirvScope, detail::getSpirvMemorySemantics(Order), Operand);
  }

  T fetch_sub(T Operand, memory_order Order = memory_order::relaxed) {
    STATIC_ASSERT_NOT_FLOAT(T);
    return ::cl::__spirv::OpAtomicISub(
        Ptr, SpirvScope, detail::getSpirvMemorySemantics(Order), Operand);
  }

  T fetch_and(T Operand, memory_order Order = memory_order::relaxed) {
    STATIC_ASSERT_NOT_FLOAT(T);
    return ::cl::__spirv::OpAtomicAnd(
        Ptr, SpirvScope, detail::getSpirvMemorySemantics(Order), Operand);
  }

  T fetch_or(T Operand, memory_order Order = memory_order::relaxed) {
    STATIC_ASSERT_NOT_FLOAT(T);
    return ::cl::__spirv::OpAtomicOr(
        Ptr, SpirvScope, detail::getSpirvMemorySemantics(Order), Operand);
  }

  T fetch_xor(T Operand, memory_order Order = memory_order::relaxed) {
    STATIC_ASSERT_NOT_FLOAT(T);
    return ::cl::__spirv::OpAtomicXor(
        Ptr, SpirvScope, detail::getSpirvMemorySemantics(Order), Operand);
  }

  T fetch_min(T Operand, memory_order Order = memory_order::relaxed) {
    STATIC_ASSERT_NOT_FLOAT(T);
    return ::cl::__spirv::OpAtomicMin(
        Ptr, SpirvScope, detail::getSpirvMemorySemantics(Order), Operand);
  }

  T fetch_max(T Operand, memory_order Order = memory_order::relaxed) {
    STATIC_ASSERT_NOT_FLOAT(T);
    return ::cl::__spirv::OpAtomicMax(
        Ptr, SpirvScope, detail::getSpirvMemorySemantics(Order), Operand);
  }

private:
#ifdef __SYCL_DEVICE_ONLY__
  typename detail::PtrValueType<T, addressSpace>::type *Ptr;
#else
  std::atomic<T> *Ptr;
#endif
};

template <typename T, access::address_space addressSpace>
void atomic_store(atomic<T, addressSpace> Object, T Operand,
                  memory_order MemoryOrder = memory_order::relaxed) {
  Object.store(Operand, MemoryOrder);
}

template <typename T, access::address_space addressSpace>
T atomic_load(atomic<T, addressSpace> Object,
              memory_order MemoryOrder = memory_order::relaxed) {
  return Object.load(MemoryOrder);
}

template <typename T, access::address_space addressSpace>
T atomic_exchange(atomic<T, addressSpace> Object, T Operand,
                  memory_order MemoryOrder = memory_order::relaxed) {
  return Object.exchange(Operand, MemoryOrder);
}

template <typename T, access::address_space addressSpace>
bool atomic_compare_exchange_strong(
    atomic<T, addressSpace> Object, T &Expected, T Desired,
    memory_order SuccessOrder = memory_order::relaxed,
    memory_order FailOrder = memory_order::relaxed) {
  return Object.compare_exchange_strong(Expected, Desired, SuccessOrder,
                                        FailOrder);
}

template <typename T, access::address_space addressSpace>
T atomic_fetch_add(atomic<T, addressSpace> Object, T Operand,
                   memory_order MemoryOrder = memory_order::relaxed) {
  return Object.fetch_add(Operand, MemoryOrder);
}

template <typename T, access::address_space addressSpace>
T atomic_fetch_sub(atomic<T, addressSpace> Object, T Operand,
                   memory_order MemoryOrder = memory_order::relaxed) {
  return Object.fetch_sub(Operand, MemoryOrder);
}

template <typename T, access::address_space addressSpace>
T atomic_fetch_and(atomic<T, addressSpace> Object, T Operand,
                   memory_order MemoryOrder = memory_order::relaxed) {
  return Object.fetch_and(Operand, MemoryOrder);
}

template <typename T, access::address_space addressSpace>
T atomic_fetch_or(atomic<T, addressSpace> Object, T Operand,
                  memory_order MemoryOrder = memory_order::relaxed) {
  return Object.fetch_or(Operand, MemoryOrder);
}

template <typename T, access::address_space addressSpace>
T atomic_fetch_xor(atomic<T, addressSpace> Object, T Operand,
                   memory_order MemoryOrder = memory_order::relaxed) {
  return Object.fetch_xor(Operand, MemoryOrder);
}

template <typename T, access::address_space addressSpace>
T atomic_fetch_min(atomic<T, addressSpace> Object, T Operand,
                   memory_order MemoryOrder = memory_order::relaxed) {
  return Object.fetch_min(Operand, MemoryOrder);
}

template <typename T, access::address_space addressSpace>
T atomic_fetch_max(atomic<T, addressSpace> Object, T Operand,
                   memory_order MemoryOrder = memory_order::relaxed) {
  return Object.fetch_max(Operand, MemoryOrder);
}

} // namespace sycl
} // namespace cl

#undef STATIC_ASSERT_NOT_FLOAT
