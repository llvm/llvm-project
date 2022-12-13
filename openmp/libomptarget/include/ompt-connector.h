//=== ompt-connector.h - Target independent OpenMP target RTL -- C++ ------===//
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

//****************************************************************************
// global includes
//****************************************************************************
#include "llvm/Support/DynamicLibrary.h"

#include <memory>
#include <string>

//****************************************************************************
// local includes
//****************************************************************************

#include <Debug.h>
#include <omp-tools.h>
#include <omptarget.h>

//****************************************************************************
// type declarations
//****************************************************************************

#define stringify(s) #s

#define LIBOMPTARGET_GET_TARGET_OPID libomptarget_get_target_opid

//****************************************************************************
// type declarations
//****************************************************************************

typedef void (*library_ompt_connect_t)(ompt_start_tool_result_t *result);

//----------------------------------------------------------------------------
// class library_ompt_connector_t
// purpose:
//
// establish connection between openmp runtime libraries
//
// NOTE: This class is intended for use in attribute constructors. therefore,
// it should be declared within the constructor function to ensure that
// the class is initialized before it's methods are used
//----------------------------------------------------------------------------

class library_ompt_connector_t {
public:
  library_ompt_connector_t(const char *ident) {
    lib_ident.append(ident);
    is_initialized = false;
  }
  library_ompt_connector_t() = delete;

  void connect(ompt_start_tool_result_t *ompt_result) {
    initialize();
    if (!library_ompt_connect)
      return;
    library_ompt_connect(ompt_result);
  }

private:
  void initialize() {
    if (is_initialized)
      return;

    std::string err_msg;
    std::string lib_name = lib_ident;
    lib_name += ".so";

    DP("OMPT: Trying to load library %s\n", lib_name.c_str());
    auto dyn_lib_handle = std::make_shared<llvm::sys::DynamicLibrary>(
        llvm::sys::DynamicLibrary::getPermanentLibrary(lib_name.c_str(),
                                                       &err_msg));
    if (!dyn_lib_handle->isValid()) {
      // The upper layer will bail out if the handle is null.
      library_ompt_connect = nullptr;
    } else {
      auto lib_conn_rtn = lib_ident + "_ompt_connect";
      DP("OMPT: Trying to get address of connection routine %s\n",
         lib_conn_rtn.c_str());
      library_ompt_connect = reinterpret_cast<library_ompt_connect_t>(
          dyn_lib_handle->getAddressOfSymbol(lib_conn_rtn.c_str()));
    }
    DP("OMPT: Library connection handle = %p\n", library_ompt_connect);
    is_initialized = true;
  }

private:
  /// Ensure initialization occurs only once
  bool is_initialized;
  /// Handle of connect routine provided by source library
  library_ompt_connect_t library_ompt_connect;
  std::string lib_ident;
};

#endif
