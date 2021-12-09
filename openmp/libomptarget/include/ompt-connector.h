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

#include <dlfcn.h>
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
  void connect(ompt_start_tool_result_t *ompt_result) {
    initialize();
    if (library_ompt_connect) {
      library_ompt_connect(ompt_result);
    }
  };

  library_ompt_connector_t(const char *library_name) {
    library_connect_routine.append(library_name);
    library_connect_routine.append("_ompt_connect");
    is_initialized = false;
  };
  library_ompt_connector_t() = delete;

private:
  void initialize() {
    if (is_initialized == false) {
      DP("OMPT: library_ompt_connect = %s\n", library_connect_routine.c_str());
      void *vptr = dlsym(NULL, library_connect_routine.c_str());
      // If dlsym fails, library_ompt_connect will be null. connect() checks
      // for this condition
      library_ompt_connect = reinterpret_cast<library_ompt_connect_t>(
          reinterpret_cast<long>(vptr));
      DP("OMPT: library_ompt_connect = %p\n", library_ompt_connect);
      is_initialized = true;
    }
  };

private:
  bool is_initialized;
  library_ompt_connect_t library_ompt_connect;
  std::string library_connect_routine;
};

#endif
