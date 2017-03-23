//===--- AMDGPUCodeObjectMetadataStreamer.h ---------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
/// \brief AMDGPU Code Object Metadata Streamer.
///
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_MCTARGETDESC_AMDGPUCODEOBJECTMETADATASTREAMER_H
#define LLVM_LIB_TARGET_AMDGPU_MCTARGETDESC_AMDGPUCODEOBJECTMETADATASTREAMER_H

#include "AMDGPUCodeObjectMetadata.h"
#include "AMDKernelCodeT.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorOr.h"

namespace llvm {

class Argument;
class DataLayout;
class FeatureBitset;
class Function;
class MDNode;
class Module;
class Type;

namespace AMDGPU {
namespace CodeObject {

class MetadataStreamer final {
private:
  Metadata CodeObjectMetadata;

  void dump(StringRef YamlString) const;

  void verify(StringRef YamlString) const;

  AccessQualifier getAccessQualifier(StringRef AccQual) const;

  AddressSpaceQualifier getAddressSpaceQualifer(unsigned AddressSpace) const;

  ValueKind getValueKind(Type *Ty, StringRef TypeQual,
                         StringRef BaseTypeName) const;

  ValueType getValueType(Type *Ty, StringRef TypeName) const;

  std::string getTypeName(Type *Ty, bool Signed) const;

  std::vector<uint32_t> getWorkGroupDimensions(MDNode *Node) const;

  void emitVersion();

  void emitIsa(const FeatureBitset &Features);

  void emitPrintf(const Module &Mod);

  void emitKernelLanguage(const Function &Func);

  void emitKernelAttrs(const Function &Func);

  void emitKernelArgs(const Function &Func);

  void emitKernelArg(const Argument &Arg);

  void emitKernelArg(const DataLayout &DL, Type *Ty, ValueKind ValueKind,
                     StringRef TypeQual = "", StringRef BaseTypeName = "",
                     StringRef AccQual = "", StringRef Name = "",
                     StringRef TypeName = "");

  void emitKernelCodeProps(const amd_kernel_code_t &KernelCode);

public:
  MetadataStreamer() = default;
  ~MetadataStreamer() = default;

  void begin(const FeatureBitset &Features, const Module &Mod);

  void end() {}

  void emitKernel(const Function &Func, const amd_kernel_code_t &KernelCode);

  ErrorOr<std::string> toYamlString();

  ErrorOr<std::string> toYamlString(const FeatureBitset &Features,
                                    StringRef YamlString);
};

} // end namespace CodeObject
} // end namespace AMDGPU
} // end namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_MCTARGETDESC_AMDGPUCODEOBJECTMETADATASTREAMER_H
