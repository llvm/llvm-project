//===- AccEntryImpl.cpp -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Logger.h"
#include "Private.h"

#include "Interface.h"
#include "include/openacc.h"

using namespace llvm::acc::target;
using namespace llvm::acc::target::debug;

#define PREAMBLE()                                                             \
  FUNC_LOGGER();                                                               \
  AccDataDescF18 *AccDataDescs[] = {nullptr};                                  \
  void *ArgPtrs[] = {Ptr};                                                     \
  void *ArgBasePtrs[] = {nullptr};                                             \
  int64_t ArgSizes[] = {static_cast<int64_t>(Bytes)};                          \
  int64_t ArgTypes[] = {TGT_ACC_MAPTYPE_NONE};

extern "C" {
int acc_is_present(void *Ptr) { return accIsPresent(Ptr); }

void *acc_create(void *Ptr, size_t Bytes) {
  return accDataEnter(nullptr, Ptr, Bytes, TGT_ACC_MAPTYPE_NONE,
                      acc_async_sync);
}
void acc_create_async(void *Ptr, size_t Bytes, int Async) {
  accDataEnter(nullptr, Ptr, Bytes, TGT_ACC_MAPTYPE_NONE, acc_async_sync);
}
void *acc_pcreate(void *Ptr, size_t Bytes) { return acc_create(Ptr, Bytes); }
void acc_pcreate_async(void *Ptr, size_t Bytes, int Async) {
  acc_create_async(Ptr, Bytes, Async);
}
void *acc_present_or_create(void *Ptr, size_t Bytes) {
  return acc_create(Ptr, Bytes);
}
void acc_present_or_create_async(void *Ptr, size_t Bytes, int Async) {
  acc_create_async(Ptr, Bytes, Async);
}

void acc_delete(void *Ptr, size_t Bytes) {
  PREAMBLE();
  __tgt_acc_data_exit(nullptr, 0, acc_device_default, 1, ArgBasePtrs, ArgPtrs,
                      ArgSizes, ArgTypes, nullptr, nullptr,
                      reinterpret_cast<AccDataDesc **>(AccDataDescs),
                      acc_async_sync);
}
void acc_delete_async(void *Ptr, size_t Bytes, int Async) {
  PREAMBLE();
  __tgt_acc_data_exit(nullptr, 0, acc_device_default, 1, ArgBasePtrs, ArgPtrs,
                      ArgSizes, ArgTypes, nullptr, nullptr,
                      reinterpret_cast<AccDataDesc **>(AccDataDescs), Async);
}
void acc_delete_finalize(void *Ptr, size_t Bytes) {
  PREAMBLE();
  ArgTypes[0] = TGT_ACC_MAPTYPE_FINALIZE;
  __tgt_acc_data_exit(nullptr, 0, acc_device_default, 1, ArgBasePtrs, ArgPtrs,
                      ArgSizes, ArgTypes, nullptr, nullptr,
                      reinterpret_cast<AccDataDesc **>(AccDataDescs),
                      acc_async_sync);
}
void acc_delete_finalize_async(void *Ptr, size_t Bytes, int Async) {
  PREAMBLE();
  ArgTypes[0] = TGT_ACC_MAPTYPE_FINALIZE;
  __tgt_acc_data_exit(nullptr, 0, acc_device_default, 1, ArgBasePtrs, ArgPtrs,
                      ArgSizes, ArgTypes, nullptr, nullptr,
                      reinterpret_cast<AccDataDesc **>(AccDataDescs), Async);
}

void *acc_copyin(void *Ptr, size_t Bytes) {
  return accDataEnter(nullptr, Ptr, Bytes, TGT_ACC_MAPTYPE_TO, acc_async_sync);
}
void acc_copyin_async(void *Ptr, size_t Bytes, int Async) {
  accDataEnter(nullptr, Ptr, Bytes, TGT_ACC_MAPTYPE_TO, acc_async_sync);
}
void *acc_pcopyin(void *Ptr, size_t Bytes) { return acc_copyin(Ptr, Bytes); }
void acc_pcopyin_async(void *Ptr, size_t Bytes, int Async) {
  acc_copyin_async(Ptr, Bytes, Async);
}
void *acc_present_or_copyin(void *Ptr, size_t Bytes) {
  return acc_copyin(Ptr, Bytes);
}
void acc_present_or_copyin_async(void *Ptr, size_t Bytes, int Async) {
  acc_copyin_async(Ptr, Bytes, Async);
}

void acc_copyout(void *Ptr, size_t Bytes) {
  PREAMBLE();
  ArgTypes[0] = TGT_ACC_MAPTYPE_FROM;
  __tgt_acc_data_exit(nullptr, 0, acc_device_default, 1, ArgBasePtrs, ArgPtrs,
                      ArgSizes, ArgTypes, nullptr, nullptr,
                      reinterpret_cast<AccDataDesc **>(AccDataDescs),
                      acc_async_sync);
}
void acc_copyout_async(void *Ptr, size_t Bytes, int Async) {
  PREAMBLE();
  ArgTypes[0] = TGT_ACC_MAPTYPE_FROM;
  __tgt_acc_data_exit(nullptr, 0, acc_device_default, 1, ArgBasePtrs, ArgPtrs,
                      ArgSizes, ArgTypes, nullptr, nullptr,
                      reinterpret_cast<AccDataDesc **>(AccDataDescs), Async);
}

void acc_copyout_finalize(void *Ptr, size_t Bytes) {
  PREAMBLE();
  ArgTypes[0] = TGT_ACC_MAPTYPE_FROM | TGT_ACC_MAPTYPE_FINALIZE;
  __tgt_acc_data_exit(nullptr, 0, acc_device_default, 1, ArgBasePtrs, ArgPtrs,
                      ArgSizes, ArgTypes, nullptr, nullptr,
                      reinterpret_cast<AccDataDesc **>(AccDataDescs),
                      acc_async_sync);
}
void acc_copyout_finalize_async(void *Ptr, size_t Bytes, int Async) {
  PREAMBLE();
  ArgTypes[0] = TGT_ACC_MAPTYPE_FROM | TGT_ACC_MAPTYPE_FINALIZE;
  __tgt_acc_data_exit(nullptr, 0, acc_device_default, 1, ArgBasePtrs, ArgPtrs,
                      ArgSizes, ArgTypes, nullptr, nullptr,
                      reinterpret_cast<AccDataDesc **>(AccDataDescs), Async);
}

void acc_update_device(void *Ptr, size_t Bytes) {
  PREAMBLE();
  ArgTypes[0] = TGT_ACC_MAPTYPE_TO;
  __tgt_acc_data_update(nullptr, 0, acc_device_default, 1, ArgBasePtrs, ArgPtrs,
                        ArgSizes, ArgTypes, nullptr, nullptr,
                        reinterpret_cast<AccDataDesc **>(AccDataDescs),
                        acc_async_sync);
}
void acc_update_device_async(void *Ptr, size_t Bytes, int Async) {
  PREAMBLE();
  ArgTypes[0] = TGT_ACC_MAPTYPE_TO;
  __tgt_acc_data_update(nullptr, 0, acc_device_default, 1, ArgBasePtrs, ArgPtrs,
                        ArgSizes, ArgTypes, nullptr, nullptr,
                        reinterpret_cast<AccDataDesc **>(AccDataDescs), Async);
}

void acc_updatein(void *Ptr, size_t Bytes) {
  PREAMBLE();
  ArgTypes[0] = TGT_ACC_MAPTYPE_TO;
  __tgt_acc_data_update(nullptr, 0, acc_device_default, 1, ArgBasePtrs, ArgPtrs,
                        ArgSizes, ArgTypes, nullptr, nullptr,
                        reinterpret_cast<AccDataDesc **>(AccDataDescs),
                        acc_async_sync);
}
void acc_updatein_async(void *Ptr, size_t Bytes, int Async) {
  PREAMBLE();
  ArgTypes[0] = TGT_ACC_MAPTYPE_TO;
  __tgt_acc_data_update(nullptr, 0, acc_device_default, 1, ArgBasePtrs, ArgPtrs,
                        ArgSizes, ArgTypes, nullptr, nullptr,
                        reinterpret_cast<AccDataDesc **>(AccDataDescs), Async);
}

void acc_update_self(void *Ptr, size_t Bytes) {
  PREAMBLE();
  ArgTypes[0] = TGT_ACC_MAPTYPE_FROM;
  __tgt_acc_data_update(nullptr, 0, acc_device_default, 1, ArgBasePtrs, ArgPtrs,
                        ArgSizes, ArgTypes, nullptr, nullptr,
                        reinterpret_cast<AccDataDesc **>(AccDataDescs),
                        acc_async_sync);
}
void acc_update_self_async(void *Ptr, size_t Bytes, int Async) {
  PREAMBLE();
  ArgTypes[0] = TGT_ACC_MAPTYPE_FROM;
  __tgt_acc_data_update(nullptr, 0, acc_device_default, 1, ArgBasePtrs, ArgPtrs,
                        ArgSizes, ArgTypes, nullptr, nullptr,
                        reinterpret_cast<AccDataDesc **>(AccDataDescs), Async);
}

void acc_update_host(void *Ptr, size_t Bytes) {
  PREAMBLE();
  ArgTypes[0] = TGT_ACC_MAPTYPE_FROM;
  __tgt_acc_data_update(nullptr, 0, acc_device_default, 1, ArgBasePtrs, ArgPtrs,
                        ArgSizes, ArgTypes, nullptr, nullptr,
                        reinterpret_cast<AccDataDesc **>(AccDataDescs),
                        acc_async_sync);
}
void acc_update_host_async(void *Ptr, size_t Bytes, int Async) {
  PREAMBLE();
  ArgTypes[0] = TGT_ACC_MAPTYPE_FROM;
  __tgt_acc_data_update(nullptr, 0, acc_device_default, 1, ArgBasePtrs, ArgPtrs,
                        ArgSizes, ArgTypes, nullptr, nullptr,
                        reinterpret_cast<AccDataDesc **>(AccDataDescs), Async);
}

void acc_updateout(void *Ptr, size_t Bytes) {
  PREAMBLE();
  ArgTypes[0] = TGT_ACC_MAPTYPE_FROM;
  __tgt_acc_data_update(nullptr, 0, acc_device_default, 1, ArgBasePtrs, ArgPtrs,
                        ArgSizes, ArgTypes, nullptr, nullptr,
                        reinterpret_cast<AccDataDesc **>(AccDataDescs),
                        acc_async_sync);
}
void acc_updateout_async(void *Ptr, size_t Bytes, int Async) {
  PREAMBLE();
  ArgTypes[0] = TGT_ACC_MAPTYPE_FROM;
  __tgt_acc_data_update(nullptr, 0, acc_device_default, 1, ArgBasePtrs, ArgPtrs,
                        ArgSizes, ArgTypes, nullptr, nullptr,
                        reinterpret_cast<AccDataDesc **>(AccDataDescs), Async);
}
}
