//===--- AMDGPUHSAMetadataStreamer.h ----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
/// AMDGPU HSA Metadata Streamer.
///
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_MCTARGETDESC_AMDGPUHSAMETADATASTREAMER_H
#define LLVM_LIB_TARGET_AMDGPU_MCTARGETDESC_AMDGPUHSAMETADATASTREAMER_H

#include "AMDGPU.h"
#include "AMDKernelCodeT.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/AMDGPUMetadata.h"

namespace llvm {

class Argument;
class DataLayout;
class Function;
class MDNode;
class Module;
struct SIProgramInfo;
class Type;

namespace AMDGPU {
namespace HSAMD {

class MetadataStreamer final {
private:
  Metadata HSAMetadata;
  AMDGPUAS AMDGPUASI;

  void dump(StringRef HSAMetadataString) const;

  void verify(StringRef HSAMetadataString) const;

  AccessQualifier getAccessQualifier(StringRef AccQual) const;

  AddressSpaceQualifier getAddressSpaceQualifer(unsigned AddressSpace) const;

  ValueKind getValueKind(Type *Ty, StringRef TypeQual,
                         StringRef BaseTypeName) const;

  ValueType getValueType(Type *Ty, StringRef TypeName) const;

  std::string getTypeName(Type *Ty, bool Signed) const;

  std::vector<uint32_t> getWorkGroupDimensions(MDNode *Node) const;

  Kernel::CodeProps::Metadata getHSACodeProps(
      const MachineFunction &MF,
      const SIProgramInfo &ProgramInfo) const;
  Kernel::DebugProps::Metadata getHSADebugProps(
      const MachineFunction &MF,
      const SIProgramInfo &ProgramInfo) const;

  void emitVersion();

  void emitPrintf(const Module &Mod);

  void emitKernelLanguage(const Function &Func);

  void emitKernelAttrs(const Function &Func);

  void emitKernelArgs(const Function &Func);

  void emitKernelArg(const Argument &Arg);

  void emitKernelArg(const DataLayout &DL, Type *Ty, ValueKind ValueKind,
                     unsigned PointeeAlign = 0,
                     StringRef Name = "", StringRef TypeName = "",
                     StringRef BaseTypeName = "", StringRef AccQual = "",
                     StringRef TypeQual = "");

  void emitHiddenKernelArgs(const Function &Func);

public:
  MetadataStreamer() = default;
  ~MetadataStreamer() = default;

  const Metadata &getHSAMetadata() const {
    return HSAMetadata;
  }

  void begin(const Module &Mod);

  void end();

  void emitKernel(const MachineFunction &MF, const SIProgramInfo &ProgramInfo);
};

} // end namespace HSAMD
} // end namespace AMDGPU
} // end namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_MCTARGETDESC_AMDGPUHSAMETADATASTREAMER_H
