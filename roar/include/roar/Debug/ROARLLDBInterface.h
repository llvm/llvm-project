//===-- ROARLLDBInterface.h - ROAR/LLDB integration glue --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Declares the ROAR/LLDB integration "glue" that allows both systems to be
// released independently. When making changes, always abide by the following:
//
//  1. make sure they are backward compatible: always add new methods to the
//     class' "tail", or create a new interface.
//  2. make sure old/legacy APIs are always supported; fail gracefully when
//     they can no longer work on new releases.
//  3. make sure the interfaces only take/return:
//     3.1) primitive types; or
//     3.2) ROARDI* types; or
//     3.3) *public* LLDB types.
//
// N.B.: it is OK to use std::unique_ptr in **non-virtual**, **inline**
//       definitions, but make sure any std::unique_ptr<T> returned by the API
//       has a matching std::default_delete<T> definition inheriting from
//       ROARDIDefaultDelete.
//
//===----------------------------------------------------------------------===//

#ifndef ROAR_DEBUG_ROARLLDBINTERFACE_H
#define ROAR_DEBUG_ROARLLDBINTERFACE_H

#include "lldb/API/LLDB.h"

#include <memory>
#include <stdlib.h>

namespace llvm {
namespace roar {
enum class Tier : uint8_t;
} // namespace roar
} // namespace llvm

namespace roar_lldb {
class ROARDebugInterface;

class ROARError {
protected:
  virtual ~ROARError();

public:
  ROARError();

  /// Destroys instance of impplementation of ROARError.
  virtual void Destroy() = 0;
  /// Sets error string.
  virtual void SetErrorString(const char *) = 0;
  /// Returns the error code.
  virtual bool Success() const = 0;
};

/// ROARDIDefaultDelete<T> is an implementation for std::default_delete<T>,
/// where T is one of the ROAR Debug Interface public types. ROARDIDefaultDelete
/// invoke the T::Destroy() method to perform instance deletion, thus ensuring
/// that the correct delete operator (i.e., the one in roar-lldb-shim.so) is
/// used to deallocate memory allocated by (roar-lldb-shim.so's) operator new.
template <typename T> struct ROARDIDefaultDelete {
  // Default-constructed.
  constexpr ROARDIDefaultDelete() = default;
  // No-op conversion between ROARDIDefaultDelete<T> and ROARDIDefaultDelete<U>.
  template <typename U>
  constexpr ROARDIDefaultDelete(const ROARDIDefaultDelete<U> &) {}
  // Frees Ptr; see implementation below for more details.
  inline void operator()(T *Ptr) const;
};
} // namespace roar_lldb

// std::default_delete<> specializations for ROAR Debug Interface public types,
// which simply inherits from ROARDIDefaultDelete<>.
template <>
struct std::default_delete<roar_lldb::ROARError>
    : roar_lldb::ROARDIDefaultDelete<roar_lldb::ROARError> {};
template <>
struct std::default_delete<roar_lldb::ROARDebugInterface>
    : roar_lldb::ROARDIDefaultDelete<roar_lldb::ROARDebugInterface> {};

namespace roar_lldb {
/// ROARDebugInterface is the main interface between LLDB and ROAR.
class ROARDebugInterface {
protected:
  ROARDebugInterface();
  virtual ~ROARDebugInterface();

public:
  static constexpr uint32_t Version = 1;

  virtual void Destroy() = 0;
  virtual uint64_t GetJitRegisterCodeAddr() = 0;
  virtual uint64_t GetDynamicSymbolArenaAddrTrigger() = 0;
  virtual const char *SetPathToRuntime(const char *Path, ROARError &Err) = 0;
  virtual uint8_t GetJITFunctionDefiningAddress(uint64_t Addr,
                                                ROARError &Err) = 0;

  /// Handle a breakpoint by address.
  virtual void HandleBreakpointByAddress(uint64_t BPAddr, void *Batton,

                                         uint8_t Adding, ROARError &Err) = 0;

  virtual void HandleBreakpointBySourceLocation(const char *SourceFilename,
                                                uint32_t LineNo, void *Batton,
                                                uint8_t Adding,

                                                ROARError &Err) = 0;

  /// Read JIT function list.
  virtual void ReadJITFunctionList(uint8_t EagerSymGen, ROARError &Err) = 0;

  /// Returns true if the given address is a trampoline function, false
  /// otherwise.
  virtual uint8_t IsFunctionTrampoline(uint64_t Addr, ROARError &Err) = 0;
  /// Notifies JIT of breakpoints that were set before symbol shared memory was
  /// initialized.
  virtual void NotifyJITWithInitSymbols(ROARError &Err) = 0;

  /// Notify a JIT that debug information needs to be loaded for this
  /// trampoline address.
  virtual void NotifyJITToLoadDebugInformation(lldb::addr_t Addr,
                                               ROARError &Err) = 0;

  /// Reset shims internal state.
  virtual void Reset() = 0;
};
} // namespace roar_lldb

/// By defining ROARDIDefaultDelete<T>::operator() here (after all other
/// roar_lldb::* definitions) T is guaranteed to have been defined, and thus
/// making the ptr->Destroy() call well-defined regardless of T.
template <typename T>
inline void roar_lldb::ROARDIDefaultDelete<T>::operator()(T *Ptr) const {
  // The deleter should gracefully ignore ptr == nullptr, thus a check must be
  // made before invoked T::Destroy() (which is virtual, thus causing a nullptr
  // dereference without the if-check).
  if (Ptr)
    Ptr->Destroy();
}

/// LLDB log function type.
typedef void (*LLDBLogFn)(const char *);
/// The factory method used for creating ROARDebugInterface objects. The
/// returned pointer is owned by the caller, and must be deleted using
/// ROARDebugInterface::Destroy (it can't be delete'd as
/// ROARDebugInterface::~ROARDebugInterface is not visible).
extern "C" roar_lldb::ROARDebugInterface *
CreateROARDebugInterface(uint32_t Version, lldb::user_id_t DebuggerID,
                         uint32_t TargetIdx, roar_lldb::ROARError &Err,
                         LLDBLogFn LLDBLog);

#endif // ROAR_DEBUG_ROARLLDBINTERFACE_H
