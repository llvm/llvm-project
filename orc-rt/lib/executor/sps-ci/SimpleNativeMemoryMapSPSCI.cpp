//===- SimpleNativeMemoryMapSPSCI.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// SPS Controller Interface implementation for SimpleNativeMemoryMap.
//
//===----------------------------------------------------------------------===//

#include "orc-rt/sps-ci/SimpleNativeMemoryMapSPSCI.h"

#include "orc-rt/SPSAllocAction.h"
#include "orc-rt/SPSMemoryFlags.h"
#include "orc-rt/SPSWrapperFunction.h"
#include "orc-rt/SimpleNativeMemoryMap.h"

namespace orc_rt {

struct SPSSimpleNativeMemoryMapSegment;

template <>
class SPSSerializationTraits<
    SPSSimpleNativeMemoryMapSegment,
    SimpleNativeMemoryMap::InitializeRequest::Segment> {
  using SPSType =
      SPSTuple<SPSAllocGroup, SPSExecutorAddr, uint64_t, SPSSequence<char>>;

public:
  static bool
  deserialize(SPSInputBuffer &IB,
              SimpleNativeMemoryMap::InitializeRequest::Segment &S) {
    AllocGroup AG;
    ExecutorAddr Address;
    uint64_t Size;
    span<const char> Content;
    if (!SPSType::AsArgList::deserialize(IB, AG, Address, Size, Content))
      return false;
    if (Size > std::numeric_limits<size_t>::max())
      return false;
    S = {AG, Address.toPtr<char *>(), static_cast<size_t>(Size), Content};
    return true;
  }
};

struct SPSSimpleNativeMemoryMapInitializeRequest;

template <>
class SPSSerializationTraits<SPSSimpleNativeMemoryMapInitializeRequest,
                             SimpleNativeMemoryMap::InitializeRequest> {
  using SPSType = SPSTuple<SPSSequence<SPSSimpleNativeMemoryMapSegment>,
                           SPSSequence<SPSAllocActionPair>>;

public:
  static bool deserialize(SPSInputBuffer &IB,
                          SimpleNativeMemoryMap::InitializeRequest &FR) {
    return SPSType::AsArgList::deserialize(IB, FR.Segments, FR.AAPs);
  }
};

namespace sps_ci {

ORC_RT_SPS_WRAPPER(
    orc_rt_SimpleNativeMemoryMap_reserve_sps_wrapper,
    SPSExpected<SPSExecutorAddr>(SPSExecutorAddr, SPSSize),
    WrapperFunction::handleWithAsyncMethod(&SimpleNativeMemoryMap::reserve))

ORC_RT_SPS_WRAPPER(orc_rt_SimpleNativeMemoryMap_releaseMultiple_sps_wrapper,
                   SPSError(SPSExecutorAddr, SPSSequence<SPSExecutorAddr>),
                   WrapperFunction::handleWithAsyncMethod(
                       &SimpleNativeMemoryMap::releaseMultiple))

ORC_RT_SPS_WRAPPER(
    orc_rt_SimpleNativeMemoryMap_initialize_sps_wrapper,
    SPSExpected<SPSExecutorAddr>(SPSExecutorAddr,
                                 SPSSimpleNativeMemoryMapInitializeRequest),
    WrapperFunction::handleWithAsyncMethod(&SimpleNativeMemoryMap::initialize))

ORC_RT_SPS_WRAPPER(
    orc_rt_SimpleNativeMemoryMap_deinitializeMultiple_sps_wrapper,
    SPSError(SPSExecutorAddr, SPSSequence<SPSExecutorAddr>),
    WrapperFunction::handleWithAsyncMethod(
        &SimpleNativeMemoryMap::deinitializeMultiple))

static std::pair<const char *, const void *>
    orc_rt_SimpleNativeMemoryMap_sps_interface[] = {
        ORC_RT_SYMTAB_PAIR(orc_rt_SimpleNativeMemoryMap_reserve_sps_wrapper),
        ORC_RT_SYMTAB_PAIR(
            orc_rt_SimpleNativeMemoryMap_releaseMultiple_sps_wrapper),
        ORC_RT_SYMTAB_PAIR(orc_rt_SimpleNativeMemoryMap_initialize_sps_wrapper),
        ORC_RT_SYMTAB_PAIR(
            orc_rt_SimpleNativeMemoryMap_deinitializeMultiple_sps_wrapper)};

Error addSimpleNativeMemoryMap(ControllerInterface &CI) {
  return CI.addSymbolsUnique(orc_rt_SimpleNativeMemoryMap_sps_interface);
}

} // namespace sps_ci
} // namespace orc_rt
