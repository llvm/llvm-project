//=== ompt_connector.h - Target independent OpenMP target RTL -- C++ ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Support used by OMPT implementation to establish communication between
// various OpenMP runtime libraries: host openmp library, target-independent
// runtime library, and device-dependent runtime libraries.
//
//===----------------------------------------------------------------------===//

#ifndef _OMPT_CONNECTOR_H
#define _OMPT_CONNECTOR_H

#ifdef OMPT_SUPPORT

#include "llvm/Support/DynamicLibrary.h"

#include <memory>
#include <string>

#include "Debug.h"
#include "omp-tools.h"
#include "omptarget.h"

#define LIBOMPTARGET_STRINGIFY(s) #s

/// Type for the function to be invoked for connecting two libraries.
typedef void (*OmptConnectRtnTy)(ompt_start_tool_result_t *result);

/// Establish connection between openmp runtime libraries
///
/// This class is used to communicate between an OMPT implementation in
/// libomptarget and libomp. It is also used to communicate between an
/// OMPT implementation in a device-specific plugin and
/// libomptarget. The decision whether OMPT is enabled or not needs to
/// be made when the library is loaded before any functions in the
/// library are invoked. For that reason, an instance of this class is
/// intended to be defined in the constructor for libomptarget or a
/// plugin so that the decision about whether OMPT is supposed to be
/// enabled is known before any interface function in the library is
/// invoked.
class OmptLibraryConnectorTy {
public:
  /// Use \p LibName as the prefix of the global function used for connecting
  /// two libraries, the source indicated by \p LibName and the destination
  /// being the one that creates this object.
  OmptLibraryConnectorTy(const char *Ident) {
    LibIdent.append(Ident);
    IsInitialized = false;
  }
  OmptLibraryConnectorTy() = delete;
  /// Use \p OmptResult init to connect the two libraries denoted by this
  /// object. The init function of \p OmptResult will be used during connection
  /// and the fini function of \p OmptResult will be used during teardown.
  void connect(ompt_start_tool_result_t *OmptResult) {
    initialize();
    if (!LibConnHandle)
      return;
    // Call the function provided by the source library for connect
    LibConnHandle(OmptResult);
  }

private:
  void initialize() {
    if (IsInitialized)
      return;

    std::string ErrMsg;
    std::string LibName = LibIdent;
    LibName += ".so";

    DP("OMPT: Trying to load library %s\n", LibName.c_str());
    auto DynLibHandle = std::make_unique<llvm::sys::DynamicLibrary>(
        llvm::sys::DynamicLibrary::getPermanentLibrary(LibName.c_str(),
                                                       &ErrMsg));
    if (!DynLibHandle->isValid()) {
      // The upper layer will bail out if the handle is null.
      LibConnHandle = nullptr;
    } else {
      auto LibConnRtn = "ompt_" + LibIdent + "_connect";
      DP("OMPT: Trying to get address of connection routine %s\n",
         LibConnRtn.c_str());
      LibConnHandle = reinterpret_cast<OmptConnectRtnTy>(
          DynLibHandle->getAddressOfSymbol(LibConnRtn.c_str()));
    }
    DP("OMPT: Library connection handle = %p\n", LibConnHandle);
    IsInitialized = true;
  }

  /// Ensure initialization occurs only once
  bool IsInitialized;
  /// Handle of connect routine provided by source library
  OmptConnectRtnTy LibConnHandle;
  /// Name of connect routine provided by source library
  std::string LibIdent;
};

#endif // OMPT_SUPPORT

#endif // _OMPT_CONNECTOR_H
