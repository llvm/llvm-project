//===- NativeDylibManagerSPSCI.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// SPS Controller Interface implementation for NativeDylibManager.
//
//===----------------------------------------------------------------------===//

#include "orc-rt/sps-ci/NativeDylibManagerSPSCI.h"
#include "orc-rt/NativeDylibManager.h"
#include "orc-rt/SPSWrapperFunction.h"

namespace orc_rt {

/// SPS serialization for NativeDylibManager::LookupFlags as a bool.
///
/// RequiredSymbol serializes as true, WeaklyReferencedSymbol as false. This
/// matches the wire format of llvm::orc::RemoteSymbolLookupSetElement, which
/// uses a bool 'Required' field.
template <>
class SPSSerializationTraits<bool, NativeDylibManager::LookupFlags> {
public:
  static size_t size(NativeDylibManager::LookupFlags) { return sizeof(bool); }

  static bool serialize(SPSOutputBuffer &OB,
                        NativeDylibManager::LookupFlags L) {
    return SPSSerializationTraits<bool, bool>::serialize(
        OB, L == NativeDylibManager::RequiredSymbol);
  }

  static bool deserialize(SPSInputBuffer &IB,
                          NativeDylibManager::LookupFlags &L) {
    bool Required;
    if (!SPSSerializationTraits<bool, bool>::deserialize(IB, Required))
      return false;
    L = Required ? NativeDylibManager::RequiredSymbol
                 : NativeDylibManager::WeaklyReferencedSymbol;
    return true;
  }
};

} // namespace orc_rt

namespace orc_rt::sps_ci {

ORC_RT_SPS_WRAPPER(
    orc_rt_ci_sps_NativeDylibManager_load,
    SPSExpected<SPSExecutorAddr>(SPSExecutorAddr, SPSString),
    WrapperFunction::handleWithAsyncMethod(&NativeDylibManager::load))

ORC_RT_SPS_WRAPPER(
    orc_rt_ci_sps_NativeDylibManager_lookup,
    SPSExpected<SPSSequence<SPSOptional<SPSExecutorAddr>>>(
        SPSExecutorAddr, SPSExecutorAddr,
        SPSSequence<SPSTuple<SPSString, bool>>),
    WrapperFunction::handleWithAsyncMethod(&NativeDylibManager::lookup))

static std::pair<const char *, const void *>
    orc_rt_ci_NativeDylibManager_sps_interface[] = {
        ORC_RT_SYMTAB_PAIR(orc_rt_ci_sps_NativeDylibManager_load),
        ORC_RT_SYMTAB_PAIR(orc_rt_ci_sps_NativeDylibManager_lookup)};

Error addNativeDylibManager(SimpleSymbolTable &ST) {
  return ST.addUnique(orc_rt_ci_NativeDylibManager_sps_interface);
}

} // namespace orc_rt::sps_ci
