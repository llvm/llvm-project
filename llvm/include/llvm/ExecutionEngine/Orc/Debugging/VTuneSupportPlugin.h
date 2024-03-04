//===--- VTuneSupportPlugin.h -- Support for VTune profiler ---*- C++ -*---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Handles support for registering code with VIntel Tune's Amplifier JIT API.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_AMPLIFIERSUPPORTPLUGIN_H
#define LLVM_EXECUTIONENGINE_ORC_AMPLIFIERSUPPORTPLUGIN_H

#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/ObjectLinkingLayer.h"

#include "llvm/ExecutionEngine/Orc/Shared/SimplePackedSerialization.h"

namespace llvm {

namespace orc {

class VTuneSupportPlugin : public ObjectLinkingLayer::Plugin {
public:
  VTuneSupportPlugin(ExecutorProcessControl &EPC, ExecutorAddr RegisterImplAddr,
                     ExecutorAddr UnregisterImplAddr, bool EmitDebugInfo)
      : EPC(EPC), RegisterVTuneImplAddr(RegisterImplAddr),
        UnregisterVTuneImplAddr(UnregisterImplAddr),
        EmitDebugInfo(EmitDebugInfo) {}

  void modifyPassConfig(MaterializationResponsibility &MR,
                        jitlink::LinkGraph &G,
                        jitlink::PassConfiguration &Config) override;

  Error notifyEmitted(MaterializationResponsibility &MR) override;
  Error notifyFailed(MaterializationResponsibility &MR) override;
  Error notifyRemovingResources(JITDylib &JD, ResourceKey K) override;
  void notifyTransferringResources(JITDylib &JD, ResourceKey DstKey,
                                   ResourceKey SrcKey) override;

  static Expected<std::unique_ptr<VTuneSupportPlugin>>
  Create(ExecutorProcessControl &EPC, JITDylib &JD, bool EmitDebugInfo,
         bool TestMode = false);

private:
  ExecutorProcessControl &EPC;
  ExecutorAddr RegisterVTuneImplAddr;
  ExecutorAddr UnregisterVTuneImplAddr;
  std::mutex PluginMutex;
  uint64_t NextMethodID{0};
  DenseMap<MaterializationResponsibility *, std::pair<uint64_t, uint64_t>>
      PendingMethodIDs;
  DenseMap<ResourceKey, std::vector<std::pair<uint64_t, uint64_t>>>
      LoadedMethodIDs;
  bool EmitDebugInfo;
};

typedef std::vector<std::pair<unsigned, unsigned>> VTuneLineTable;

// SI = String Index, 1-indexed into the VTuneMethodBatch::Strings table.
// SI == 0 means replace with nullptr.

// MI = Method Index, 1-indexed into the VTuneMethodBatch::Methods table.
// MI == 0 means this is a parent method and was not inlined.

struct VTuneMethodInfo {
  VTuneLineTable LineTable;
  ExecutorAddr LoadAddr;
  uint64_t LoadSize;
  uint64_t MethodID;
  uint32_t NameSI;
  uint32_t ClassFileSI;
  uint32_t SourceFileSI;
  uint32_t ParentMI;
};

typedef std::vector<VTuneMethodInfo> VTuneMethodTable;
typedef std::vector<std::string> VTuneStringTable;

struct VTuneMethodBatch {
  VTuneMethodTable Methods;
  VTuneStringTable Strings;
};

typedef std::vector<std::pair<uint64_t, uint64_t>> VTuneUnloadedMethodIDs;

namespace shared {

using SPSVTuneLineTable = SPSSequence<SPSTuple<uint32_t, uint32_t>>;
using SPSVTuneMethodInfo =
    SPSTuple<SPSVTuneLineTable, SPSExecutorAddr, uint64_t, uint64_t, uint32_t,
             uint32_t, uint32_t, uint32_t>;
using SPSVTuneMethodTable = SPSSequence<SPSVTuneMethodInfo>;
using SPSVTuneStringTable = SPSSequence<SPSString>;
using SPSVTuneMethodBatch = SPSTuple<SPSVTuneMethodTable, SPSVTuneStringTable>;
using SPSVTuneUnloadedMethodIDs = SPSSequence<SPSTuple<uint64_t, uint64_t>>;

template <> class SPSSerializationTraits<SPSVTuneMethodInfo, VTuneMethodInfo> {
public:
  static size_t size(const VTuneMethodInfo &MI) {
    return SPSVTuneMethodInfo::AsArgList::size(
        MI.LineTable, MI.LoadAddr, MI.LoadSize, MI.MethodID, MI.NameSI,
        MI.ClassFileSI, MI.SourceFileSI, MI.ParentMI);
  }

  static bool deserialize(SPSInputBuffer &IB, VTuneMethodInfo &MI) {
    return SPSVTuneMethodInfo::AsArgList::deserialize(
        IB, MI.LineTable, MI.LoadAddr, MI.LoadSize, MI.MethodID, MI.NameSI,
        MI.ClassFileSI, MI.SourceFileSI, MI.ParentMI);
  }

  static bool serialize(SPSOutputBuffer &OB, const VTuneMethodInfo &MI) {
    return SPSVTuneMethodInfo::AsArgList::serialize(
        OB, MI.LineTable, MI.LoadAddr, MI.LoadSize, MI.MethodID, MI.NameSI,
        MI.ClassFileSI, MI.SourceFileSI, MI.ParentMI);
  }
};

template <>
class SPSSerializationTraits<SPSVTuneMethodBatch, VTuneMethodBatch> {
public:
  static size_t size(const VTuneMethodBatch &MB) {
    return SPSVTuneMethodBatch::AsArgList::size(MB.Methods, MB.Strings);
  }

  static bool deserialize(SPSInputBuffer &IB, VTuneMethodBatch &MB) {
    return SPSVTuneMethodBatch::AsArgList::deserialize(IB, MB.Methods,
                                                       MB.Strings);
  }

  static bool serialize(SPSOutputBuffer &OB, const VTuneMethodBatch &MB) {
    return SPSVTuneMethodBatch::AsArgList::serialize(OB, MB.Methods,
                                                     MB.Strings);
  }
};

} // end namespace shared

} // end namespace orc

} // end namespace llvm

#endif
