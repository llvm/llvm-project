//===-- FortranInterface.cpp - Fortran OpenMP API entry points --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Fortran entry points for OpenMP API routines on the device.
//
// These entry points use the default Flang name mangling. omp_lib cannot
// declare the routines BIND(C), because their OPTIONAL dummy arguments cannot
// also have the VALUE attribute in BIND(C) routines. Flang therefore passes
// all arguments by reference, and an absent OPTIONAL argument is passed as a
// null pointer. Each entry point substitutes the default for absent arguments
// and forwards to the corresponding C API routine.
//
// TODO: If the name mangling changes or is different for other OSes,
//       implement improved name mangling schemes.
//
//===----------------------------------------------------------------------===//

#include "DeviceTypes.h"
#include "Interface.h"

extern "C" {

void *omp_get_dyn_gprivate_ptr_(size_t *Offset, omp_access_t *AccessGroup) {
  size_t Off = Offset ? *Offset : 0;
  omp_access_t AccGrp = AccessGroup ? *AccessGroup : omp_access_cgroup;
  return omp_get_dyn_gprivate_ptr(Off, AccGrp);
}

void *omp_get_dyn_gprivate_nofb_ptr_(size_t *Offset,
                                     omp_access_t *AccessGroup) {
  size_t Off = Offset ? *Offset : 0;
  omp_access_t AccGrp = AccessGroup ? *AccessGroup : omp_access_cgroup;
  return omp_get_dyn_gprivate_nofb_ptr(Off, AccGrp);
}

size_t omp_get_dyn_gprivate_size_(omp_access_t *AccessGroup) {
  omp_access_t AccGrp = AccessGroup ? *AccessGroup : omp_access_cgroup;
  return omp_get_dyn_gprivate_size(AccGrp);
}

omp_memspace_handle_t
omp_get_dyn_gprivate_memspace_(omp_access_t *AccessGroup) {
  omp_access_t AccGrp = AccessGroup ? *AccessGroup : omp_access_cgroup;
  return omp_get_dyn_gprivate_memspace(AccGrp);
}

} // extern "C"
