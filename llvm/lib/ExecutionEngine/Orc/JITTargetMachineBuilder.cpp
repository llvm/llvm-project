//===----- JITTargetMachineBuilder.cpp - Build TargetMachines for JIT -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"

#include "llvm/Support/Host.h"
#include "llvm/Support/TargetRegistry.h"

namespace llvm {
namespace orc {

JITTargetMachineBuilder::JITTargetMachineBuilder(Triple TT)
    : TT(std::move(TT)) {
  Options.EmulatedTLS = true;
  Options.ExplicitEmulatedTLS = true;
}

Expected<JITTargetMachineBuilder> JITTargetMachineBuilder::detectHost() {
  // FIXME: getProcessTriple is bogus. It returns the host LLVM was compiled on,
  //        rather than a valid triple for the current process.
  JITTargetMachineBuilder TMBuilder((Triple(sys::getProcessTriple())));

  // Retrieve host CPU name and sub-target features and add them to builder.
  // Relocation model, code model and codegen opt level are kept to default
  // values.
  llvm::StringMap<bool> FeatureMap;
  llvm::sys::getHostCPUFeatures(FeatureMap);
  for (auto &Feature : FeatureMap)
    TMBuilder.getFeatures().AddFeature(Feature.first(), Feature.second);

  TMBuilder.setCPU(llvm::sys::getHostCPUName());

  return TMBuilder;
}

Expected<std::unique_ptr<TargetMachine>>
JITTargetMachineBuilder::createTargetMachine() {

  std::string ErrMsg;
  auto *TheTarget = TargetRegistry::lookupTarget(TT.getTriple(), ErrMsg);
  if (!TheTarget)
    return make_error<StringError>(std::move(ErrMsg), inconvertibleErrorCode());

  auto *TM =
      TheTarget->createTargetMachine(TT.getTriple(), CPU, Features.getString(),
                                     Options, RM, CM, OptLevel, /*JIT*/ true);
  if (!TM)
    return make_error<StringError>("Could not allocate target machine",
                                   inconvertibleErrorCode());

  return std::unique_ptr<TargetMachine>(TM);
}

JITTargetMachineBuilder &JITTargetMachineBuilder::addFeatures(
    const std::vector<std::string> &FeatureVec) {
  for (const auto &F : FeatureVec)
    Features.AddFeature(F);
  return *this;
}

} // End namespace orc.
} // End namespace llvm.
