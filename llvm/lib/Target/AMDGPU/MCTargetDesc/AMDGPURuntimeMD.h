//===- AMDGPURuntimeMD.h - Generate runtime metadata ---------------*- C++ -*-//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares functions for generating runtime metadata.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_MCTARGETDESC_AMDGPURUNTIMEMD_H
#define LLVM_LIB_TARGET_AMDGPU_MCTARGETDESC_AMDGPURUNTIMEMD_H

#include "llvm/Support/ErrorOr.h"
#include <string>

namespace llvm {
class FeatureBitset;
class Module;

/// \returns Runtime metadata as YAML string.
std::string getRuntimeMDYAMLString(const FeatureBitset &Features,
                                   const Module &M);

/// \returns \p YAML if \p YAML is valid runtime metadata, error otherwise.
ErrorOr<std::string> getRuntimeMDYAMLString(const FeatureBitset &Features,
                                            StringRef YAML);

}
#endif
