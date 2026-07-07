//===- MemoryAccessSPSCI.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// SPS Controller Interface implementation for MemoryAccess.
//
//===----------------------------------------------------------------------===//

#include "orc-rt/sps-ci/MemoryAccessSPSCI.h"
#include "orc-rt/SPSWrapperFunction.h"
#include "orc-rt/move_only_function.h"

#include <cstring>
#include <vector>

using namespace orc_rt;

namespace {

template <typename T>
void writePrimitives(move_only_function<void()> &&OnComplete,
                     std::vector<std::pair<T *, T>> Writes) {
  for (auto &[Ptr, Value] : Writes)
    *Ptr = Value;
  OnComplete();
}

void writeBuffers(move_only_function<void()> &&OnComplete,
                  std::vector<std::pair<char *, std::vector<char>>> Writes) {
  for (auto &[Ptr, Value] : Writes)
    memcpy(Ptr, Value.data(), Value.size());
  OnComplete();
}

template <typename T>
void readPrimitives(move_only_function<void(std::vector<T>)> &&OnComplete,
                    std::vector<const T *> Reads) {
  std::vector<T> Values;
  Values.reserve(Reads.size());
  for (auto *Ptr : Reads)
    Values.push_back(*Ptr);
  OnComplete(std::move(Values));
}

void readBuffers(
    move_only_function<void(std::vector<std::vector<char>>)> &&OnComplete,
    std::vector<std::pair<const char *, uint64_t>> Reads) {

  std::vector<std::vector<char>> Values;
  Values.reserve(Reads.size());
  for (auto &[Ptr, Size] : Reads) {
    Values.push_back({});
    Values.back().resize(Size);
    memcpy(Values.back().data(), Ptr, Size);
  }
  OnComplete(std::move(Values));
}

void readStrings(
    move_only_function<void(std::vector<std::string>)> &&OnComplete,
    std::vector<const char *> Reads) {
  std::vector<std::string> Values;
  Values.reserve(Reads.size());
  for (auto *Ptr : Reads) {
    Values.push_back({});
    while (*Ptr != '\0')
      Values.back().push_back(*Ptr++);
  }
  OnComplete(std::move(Values));
}

} // anonymous namespace
namespace orc_rt::sps_ci {

ORC_RT_SPS_WRAPPER(orc_rt_ci_sps_mem_write_uint8s,
                   void(SPSSequence<SPSTuple<SPSExecutorAddr, uint8_t>>),
                   writePrimitives<uint8_t>);

ORC_RT_SPS_WRAPPER(orc_rt_ci_sps_mem_write_uint16s,
                   void(SPSSequence<SPSTuple<SPSExecutorAddr, uint16_t>>),
                   writePrimitives<uint16_t>);

ORC_RT_SPS_WRAPPER(orc_rt_ci_sps_mem_write_uint32s,
                   void(SPSSequence<SPSTuple<SPSExecutorAddr, uint32_t>>),
                   writePrimitives<uint32_t>);

ORC_RT_SPS_WRAPPER(orc_rt_ci_sps_mem_write_uint64s,
                   void(SPSSequence<SPSTuple<SPSExecutorAddr, uint64_t>>),
                   writePrimitives<uint64_t>);

ORC_RT_SPS_WRAPPER(
    orc_rt_ci_sps_mem_write_pointers,
    void(SPSSequence<SPSTuple<SPSExecutorAddr, SPSExecutorAddr>>),
    writePrimitives<void *>);

ORC_RT_SPS_WRAPPER(
    orc_rt_ci_sps_mem_write_buffers,
    void(SPSSequence<SPSTuple<SPSExecutorAddr, SPSSequence<char>>>),
    writeBuffers);

ORC_RT_SPS_WRAPPER(orc_rt_ci_sps_mem_read_uint8s,
                   SPSSequence<uint8_t>(SPSSequence<SPSExecutorAddr>),
                   readPrimitives<uint8_t>);

ORC_RT_SPS_WRAPPER(orc_rt_ci_sps_mem_read_uint16s,
                   SPSSequence<uint16_t>(SPSSequence<SPSExecutorAddr>),
                   readPrimitives<uint16_t>);

ORC_RT_SPS_WRAPPER(orc_rt_ci_sps_mem_read_uint32s,
                   SPSSequence<uint32_t>(SPSSequence<SPSExecutorAddr>),
                   readPrimitives<uint32_t>);

ORC_RT_SPS_WRAPPER(orc_rt_ci_sps_mem_read_uint64s,
                   SPSSequence<uint64_t>(SPSSequence<SPSExecutorAddr>),
                   readPrimitives<uint64_t>);

ORC_RT_SPS_WRAPPER(orc_rt_ci_sps_mem_read_pointers,
                   SPSSequence<SPSExecutorAddr>(SPSSequence<SPSExecutorAddr>),
                   readPrimitives<void *>);

ORC_RT_SPS_WRAPPER(orc_rt_ci_sps_mem_read_buffers,
                   SPSSequence<SPSSequence<char>>(
                       SPSSequence<SPSTuple<SPSExecutorAddr, uint64_t>>),
                   readBuffers);

ORC_RT_SPS_WRAPPER(orc_rt_ci_sps_mem_read_strings,
                   SPSSequence<SPSString>(SPSSequence<SPSExecutorAddr>),
                   readStrings);

static std::pair<const char *, const void *>
    orc_rt_ci_MemoryAccess_sps_interface[] = {
        ORC_RT_SYMTAB_PAIR(orc_rt_ci_sps_mem_write_uint8s),
        ORC_RT_SYMTAB_PAIR(orc_rt_ci_sps_mem_write_uint16s),
        ORC_RT_SYMTAB_PAIR(orc_rt_ci_sps_mem_write_uint32s),
        ORC_RT_SYMTAB_PAIR(orc_rt_ci_sps_mem_write_uint64s),
        ORC_RT_SYMTAB_PAIR(orc_rt_ci_sps_mem_write_pointers),
        ORC_RT_SYMTAB_PAIR(orc_rt_ci_sps_mem_write_buffers),
        ORC_RT_SYMTAB_PAIR(orc_rt_ci_sps_mem_read_uint8s),
        ORC_RT_SYMTAB_PAIR(orc_rt_ci_sps_mem_read_uint16s),
        ORC_RT_SYMTAB_PAIR(orc_rt_ci_sps_mem_read_uint32s),
        ORC_RT_SYMTAB_PAIR(orc_rt_ci_sps_mem_read_uint64s),
        ORC_RT_SYMTAB_PAIR(orc_rt_ci_sps_mem_read_pointers),
        ORC_RT_SYMTAB_PAIR(orc_rt_ci_sps_mem_read_buffers),
        ORC_RT_SYMTAB_PAIR(orc_rt_ci_sps_mem_read_strings)};

Error addMemoryAccess(SimpleSymbolTable &ST) {
  return ST.addUnique(orc_rt_ci_MemoryAccess_sps_interface);
}

} // namespace orc_rt::sps_ci
