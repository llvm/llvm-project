//===- CfiAccEntryImpl.cpp --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DeviceManager.h"
#include "Logger.h"
#include "Private.h"
#include "omptarget.h"

#include "Interface.h"
#include "include/openacc.h"

using namespace llvm::acc::target;
using namespace llvm::acc::target::debug;

#define PREAMBLE()                                                             \
  FUNC_LOGGER();                                                               \
  AccDataDescF18 AccDesc{{TGT_ACC_DESC_F18}, &Desc->raw()};                    \
  AccDataDescF18 *AccDataDescs[] = {&AccDesc};                                 \
  void *ArgPtrs[] = {reinterpret_cast<void *>(&Desc->raw())};                  \
  void *ArgBasePtrs[] = {nullptr};                                             \
  int64_t ArgSizes[] = {0};                                                    \
  int64_t ArgTypes[] = {TGT_ACC_MAPTYPE_NONE};

extern "C" {
int _cfi_acc_is_present_a(const Fortran::runtime::Descriptor *Desc) {
  return accIsPresent(Desc->OffsetElement());
}

int _cfi_acc_create_a(Fortran::runtime::Descriptor *Desc) {
  PREAMBLE();
  __tgt_acc_data_enter(nullptr, 0, acc_device_default, 1, ArgBasePtrs, ArgPtrs,
                       ArgSizes, ArgTypes, nullptr, nullptr,
                       reinterpret_cast<AccDataDesc **>(AccDataDescs),
                       acc_async_sync);
  return 0;
}
int _cfi_acc_create_async_a(Fortran::runtime::Descriptor *Desc, int *Async) {
  PREAMBLE();
  __tgt_acc_data_enter(nullptr, 0, acc_device_default, 1, ArgBasePtrs, ArgPtrs,
                       ArgSizes, ArgTypes, nullptr, nullptr,
                       reinterpret_cast<AccDataDesc **>(AccDataDescs), *Async);
  return 0;
}
int _cfi_acc_pcreate_a(Fortran::runtime::Descriptor *Desc) {
  PREAMBLE();
  __tgt_acc_data_enter(nullptr, 0, acc_device_default, 1, ArgBasePtrs, ArgPtrs,
                       ArgSizes, ArgTypes, nullptr, nullptr,
                       reinterpret_cast<AccDataDesc **>(AccDataDescs),
                       acc_async_sync);
  return 0;
}
int _cfi_acc_pcreate_async_a(Fortran::runtime::Descriptor *Desc, int *Async) {
  PREAMBLE();
  __tgt_acc_data_enter(nullptr, 0, acc_device_default, 1, ArgBasePtrs, ArgPtrs,
                       ArgSizes, ArgTypes, nullptr, nullptr,
                       reinterpret_cast<AccDataDesc **>(AccDataDescs), *Async);
  return 0;
}
int _cfi_acc_present_or_create_a(Fortran::runtime::Descriptor *Desc) {
  PREAMBLE();
  __tgt_acc_data_enter(nullptr, 0, acc_device_default, 1, ArgBasePtrs, ArgPtrs,
                       ArgSizes, ArgTypes, nullptr, nullptr,
                       reinterpret_cast<AccDataDesc **>(AccDataDescs),
                       acc_async_sync);
  return 0;
}
int _cfi_acc_present_or_create_async_a(Fortran::runtime::Descriptor *Desc,
                                       int *Async) {
  PREAMBLE();
  __tgt_acc_data_enter(nullptr, 0, acc_device_default, 1, ArgBasePtrs, ArgPtrs,
                       ArgSizes, ArgTypes, nullptr, nullptr,
                       reinterpret_cast<AccDataDesc **>(AccDataDescs), *Async);
  return 0;
}

int _cfi_acc_delete_a(Fortran::runtime::Descriptor *Desc) {
  PREAMBLE();
  __tgt_acc_data_exit(nullptr, 0, acc_device_default, 1, ArgBasePtrs, ArgPtrs,
                      ArgSizes, ArgTypes, nullptr, nullptr,
                      reinterpret_cast<AccDataDesc **>(AccDataDescs),
                      acc_async_sync);
  return 0;
}
int _cfi_acc_delete_async_a(Fortran::runtime::Descriptor *Desc, int *Async) {
  PREAMBLE();
  __tgt_acc_data_exit(nullptr, 0, acc_device_default, 1, ArgBasePtrs, ArgPtrs,
                      ArgSizes, ArgTypes, nullptr, nullptr,
                      reinterpret_cast<AccDataDesc **>(AccDataDescs), *Async);
  return 0;
}
int _cfi_acc_delete_finalize_a(Fortran::runtime::Descriptor *Desc) {
  PREAMBLE();
  ArgTypes[0] = TGT_ACC_MAPTYPE_FINALIZE;
  __tgt_acc_data_exit(nullptr, 0, acc_device_default, 1, ArgBasePtrs, ArgPtrs,
                      ArgSizes, ArgTypes, nullptr, nullptr,
                      reinterpret_cast<AccDataDesc **>(AccDataDescs),
                      acc_async_sync);
  return 0;
}
int _cfi_acc_delete_finalize_async_a(Fortran::runtime::Descriptor *Desc,
                                     int *Async) {
  PREAMBLE();
  ArgTypes[0] = TGT_ACC_MAPTYPE_FINALIZE;
  __tgt_acc_data_exit(nullptr, 0, acc_device_default, 1, ArgBasePtrs, ArgPtrs,
                      ArgSizes, ArgTypes, nullptr, nullptr,
                      reinterpret_cast<AccDataDesc **>(AccDataDescs), *Async);
  return 0;
}

int _cfi_acc_copyin_a(Fortran::runtime::Descriptor *Desc) {
  PREAMBLE();
  ArgTypes[0] = TGT_ACC_MAPTYPE_TO;
  __tgt_acc_data_enter(nullptr, 0, acc_device_default, 1, ArgBasePtrs, ArgPtrs,
                       ArgSizes, ArgTypes, nullptr, nullptr,
                       reinterpret_cast<AccDataDesc **>(AccDataDescs),
                       acc_async_sync);
  return 0;
}
int _cfi_acc_copyin_async_a(Fortran::runtime::Descriptor *Desc, int *Async) {
  PREAMBLE();
  ArgTypes[0] = TGT_ACC_MAPTYPE_TO;
  __tgt_acc_data_enter(nullptr, 0, acc_device_default, 1, ArgBasePtrs, ArgPtrs,
                       ArgSizes, ArgTypes, nullptr, nullptr,
                       reinterpret_cast<AccDataDesc **>(AccDataDescs), *Async);
  return 0;
}
int _cfi_acc_pcopyin_a(Fortran::runtime::Descriptor *Desc) {
  PREAMBLE();
  ArgTypes[0] = TGT_ACC_MAPTYPE_TO;
  __tgt_acc_data_enter(nullptr, 0, acc_device_default, 1, ArgBasePtrs, ArgPtrs,
                       ArgSizes, ArgTypes, nullptr, nullptr,
                       reinterpret_cast<AccDataDesc **>(AccDataDescs),
                       acc_async_sync);
  return 0;
}
int _cfi_acc_pcopyin_async_a(Fortran::runtime::Descriptor *Desc, int *Async) {
  PREAMBLE();
  ArgTypes[0] = TGT_ACC_MAPTYPE_TO;
  __tgt_acc_data_enter(nullptr, 0, acc_device_default, 1, ArgBasePtrs, ArgPtrs,
                       ArgSizes, ArgTypes, nullptr, nullptr,
                       reinterpret_cast<AccDataDesc **>(AccDataDescs), *Async);
  return 0;
}
int _cfi_acc_present_or_copyin_a(Fortran::runtime::Descriptor *Desc) {
  PREAMBLE();
  ArgTypes[0] = TGT_ACC_MAPTYPE_TO;
  __tgt_acc_data_enter(nullptr, 0, acc_device_default, 1, ArgBasePtrs, ArgPtrs,
                       ArgSizes, ArgTypes, nullptr, nullptr,
                       reinterpret_cast<AccDataDesc **>(AccDataDescs),
                       acc_async_sync);
  return 0;
}
int _cfi_acc_present_or_copyin_async_a(Fortran::runtime::Descriptor *Desc,
                                       int *Async) {
  PREAMBLE();
  ArgTypes[0] = TGT_ACC_MAPTYPE_TO;
  __tgt_acc_data_enter(nullptr, 0, acc_device_default, 1, ArgBasePtrs, ArgPtrs,
                       ArgSizes, ArgTypes, nullptr, nullptr,
                       reinterpret_cast<AccDataDesc **>(AccDataDescs), *Async);
  return 0;
}

int _cfi_acc_copyout_a(Fortran::runtime::Descriptor *Desc) {
  PREAMBLE();
  ArgTypes[0] = TGT_ACC_MAPTYPE_FROM;
  __tgt_acc_data_exit(nullptr, 0, acc_device_default, 1, ArgBasePtrs, ArgPtrs,
                      ArgSizes, ArgTypes, nullptr, nullptr,
                      reinterpret_cast<AccDataDesc **>(AccDataDescs),
                      acc_async_sync);
  return 0;
}
int _cfi_acc_copyout_async_a(Fortran::runtime::Descriptor *Desc, int *Async) {
  PREAMBLE();
  ArgTypes[0] = TGT_ACC_MAPTYPE_FROM;
  __tgt_acc_data_exit(nullptr, 0, acc_device_default, 1, ArgBasePtrs, ArgPtrs,
                      ArgSizes, ArgTypes, nullptr, nullptr,
                      reinterpret_cast<AccDataDesc **>(AccDataDescs), *Async);
  return 0;
}

int _cfi_acc_copyout_finalize_a(Fortran::runtime::Descriptor *Desc) {
  PREAMBLE();
  ArgTypes[0] = TGT_ACC_MAPTYPE_FROM | TGT_ACC_MAPTYPE_FINALIZE;
  __tgt_acc_data_exit(nullptr, 0, acc_device_default, 1, ArgBasePtrs, ArgPtrs,
                      ArgSizes, ArgTypes, nullptr, nullptr,
                      reinterpret_cast<AccDataDesc **>(AccDataDescs),
                      acc_async_sync);
  return 0;
}
int _cfi_acc_copyout_finalize_async_a(Fortran::runtime::Descriptor *Desc,
                                      int *Async) {
  PREAMBLE();
  ArgTypes[0] = TGT_ACC_MAPTYPE_FROM | TGT_ACC_MAPTYPE_FINALIZE;
  __tgt_acc_data_exit(nullptr, 0, acc_device_default, 1, ArgBasePtrs, ArgPtrs,
                      ArgSizes, ArgTypes, nullptr, nullptr,
                      reinterpret_cast<AccDataDesc **>(AccDataDescs), *Async);
  return 0;
}

int _cfi_acc_update_device_a(Fortran::runtime::Descriptor *Desc) {
  PREAMBLE();
  ArgTypes[0] = TGT_ACC_MAPTYPE_TO;
  __tgt_acc_data_update(nullptr, 0, acc_device_default, 1, ArgBasePtrs, ArgPtrs,
                        ArgSizes, ArgTypes, nullptr, nullptr,
                        reinterpret_cast<AccDataDesc **>(AccDataDescs),
                        acc_async_sync);
  return 0;
}
int _cfi_acc_update_device_async_a(Fortran::runtime::Descriptor *Desc,
                                   int *Async) {
  PREAMBLE();
  ArgTypes[0] = TGT_ACC_MAPTYPE_TO;
  __tgt_acc_data_update(nullptr, 0, acc_device_default, 1, ArgBasePtrs, ArgPtrs,
                        ArgSizes, ArgTypes, nullptr, nullptr,
                        reinterpret_cast<AccDataDesc **>(AccDataDescs), *Async);
  return 0;
}

int _cfi_acc_updatein_a(Fortran::runtime::Descriptor *Desc) {
  PREAMBLE();
  ArgTypes[0] = TGT_ACC_MAPTYPE_TO;
  __tgt_acc_data_update(nullptr, 0, acc_device_default, 1, ArgBasePtrs, ArgPtrs,
                        ArgSizes, ArgTypes, nullptr, nullptr,
                        reinterpret_cast<AccDataDesc **>(AccDataDescs),
                        acc_async_sync);
  return 0;
}
int _cfi_acc_updatein_async_a(Fortran::runtime::Descriptor *Desc, int *Async) {
  PREAMBLE();
  ArgTypes[0] = TGT_ACC_MAPTYPE_TO;
  __tgt_acc_data_update(nullptr, 0, acc_device_default, 1, ArgBasePtrs, ArgPtrs,
                        ArgSizes, ArgTypes, nullptr, nullptr,
                        reinterpret_cast<AccDataDesc **>(AccDataDescs), *Async);
  return 0;
}

int _cfi_acc_update_self_a(Fortran::runtime::Descriptor *Desc) {
  PREAMBLE();
  ArgTypes[0] = TGT_ACC_MAPTYPE_FROM;
  __tgt_acc_data_update(nullptr, 0, acc_device_default, 1, ArgBasePtrs, ArgPtrs,
                        ArgSizes, ArgTypes, nullptr, nullptr,
                        reinterpret_cast<AccDataDesc **>(AccDataDescs),
                        acc_async_sync);
  return 0;
}
int _cfi_acc_update_self_async_a(Fortran::runtime::Descriptor *Desc,
                                 int *Async) {
  PREAMBLE();
  ArgTypes[0] = TGT_ACC_MAPTYPE_FROM;
  __tgt_acc_data_update(nullptr, 0, acc_device_default, 1, ArgBasePtrs, ArgPtrs,
                        ArgSizes, ArgTypes, nullptr, nullptr,
                        reinterpret_cast<AccDataDesc **>(AccDataDescs), *Async);
  return 0;
}

int _cfi_acc_update_host_a(Fortran::runtime::Descriptor *Desc) {
  PREAMBLE();
  ArgTypes[0] = TGT_ACC_MAPTYPE_FROM;
  __tgt_acc_data_update(nullptr, 0, acc_device_default, 1, ArgBasePtrs, ArgPtrs,
                        ArgSizes, ArgTypes, nullptr, nullptr,
                        reinterpret_cast<AccDataDesc **>(AccDataDescs),
                        acc_async_sync);
  return 0;
}
int _cfi_acc_update_host_async_a(Fortran::runtime::Descriptor *Desc,
                                 int *Async) {
  PREAMBLE();
  ArgTypes[0] = TGT_ACC_MAPTYPE_FROM;
  __tgt_acc_data_update(nullptr, 0, acc_device_default, 1, ArgBasePtrs, ArgPtrs,
                        ArgSizes, ArgTypes, nullptr, nullptr,
                        reinterpret_cast<AccDataDesc **>(AccDataDescs), *Async);
  return 0;
}

int _cfi_acc_updateout_a(Fortran::runtime::Descriptor *Desc) {
  PREAMBLE();
  ArgTypes[0] = TGT_ACC_MAPTYPE_FROM;
  __tgt_acc_data_update(nullptr, 0, acc_device_default, 1, ArgBasePtrs, ArgPtrs,
                        ArgSizes, ArgTypes, nullptr, nullptr,
                        reinterpret_cast<AccDataDesc **>(AccDataDescs),
                        acc_async_sync);
  return 0;
}
int _cfi_acc_updateout_async_a(Fortran::runtime::Descriptor *Desc, int *Async) {
  PREAMBLE();
  ArgTypes[0] = TGT_ACC_MAPTYPE_FROM;
  __tgt_acc_data_update(nullptr, 0, acc_device_default, 1, ArgBasePtrs, ArgPtrs,
                        ArgSizes, ArgTypes, nullptr, nullptr,
                        reinterpret_cast<AccDataDesc **>(AccDataDescs), *Async);
  return 0;
}
}
