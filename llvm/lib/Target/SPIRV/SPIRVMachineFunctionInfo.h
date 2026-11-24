//===-- SPIRVMachineFunctionInfo.h - SPIR-V machine function info -*- C++ -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// MIR serialization of the SPIR-V type assigned to each virtual register.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_SPIRV_SPIRVMACHINEFUNCTIONINFO_H
#define LLVM_LIB_TARGET_SPIRV_SPIRVMACHINEFUNCTIONINFO_H

#include "llvm/CodeGen/MIRYamlMapping.h"

namespace llvm {
class MachineFunction;

namespace yaml {

struct SPIRVVRegType {
  StringValue Reg;
  StringValue Type;
};

template <> struct MappingTraits<SPIRVVRegType> {
  static void mapping(IO &YamlIO, SPIRVVRegType &VRegType) {
    YamlIO.mapRequired("reg", VRegType.Reg);
    YamlIO.mapRequired("type", VRegType.Type);
  }

  static const bool flow = true;
};

} // end namespace yaml
} // end namespace llvm

LLVM_YAML_IS_SEQUENCE_VECTOR(llvm::yaml::SPIRVVRegType)

namespace llvm {
namespace yaml {

struct SPIRVMachineFunctionInfo final : public yaml::MachineFunctionInfo {
  std::vector<SPIRVVRegType> VRegTypes;

  SPIRVMachineFunctionInfo() = default;
  SPIRVMachineFunctionInfo(const llvm::MachineFunction &MF);

  void mappingImpl(yaml::IO &YamlIO) override;
};

template <> struct MappingTraits<SPIRVMachineFunctionInfo> {
  static void mapping(IO &YamlIO, SPIRVMachineFunctionInfo &MFI) {
    YamlIO.mapOptional("vregTypes", MFI.VRegTypes);
  }
};

} // end namespace yaml
} // end namespace llvm

#endif // LLVM_LIB_TARGET_SPIRV_SPIRVMACHINEFUNCTIONINFO_H
