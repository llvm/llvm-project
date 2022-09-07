//===--- AMDGPUHSAMetadataStreamer.h ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

#include "llvm/BinaryFormat/MsgPackDocument.h"
#include "llvm/Support/AMDGPUMetadata.h"
#include "llvm/Support/Alignment.h"

namespace llvm {

class AMDGPUTargetStreamer;
class Argument;
class DataLayout;
class Function;
class MachineFunction;
class MDNode;
class Module;
struct SIProgramInfo;
class Type;
class GCNSubtarget;

namespace AMDGPU {

namespace IsaInfo {
class AMDGPUTargetID;
}

namespace HSAMD {

class MetadataStreamer {
public:
  virtual ~MetadataStreamer() = default;

  virtual bool emitTo(AMDGPUTargetStreamer &TargetStreamer) = 0;

  virtual void begin(const Module &Mod,
                     const IsaInfo::AMDGPUTargetID &TargetID) = 0;

  virtual void end() = 0;

  virtual void emitKernel(const MachineFunction &MF,
                          const SIProgramInfo &ProgramInfo) = 0;

protected:
  virtual void emitVersion() = 0;
  virtual void emitHiddenKernelArgs(const MachineFunction &MF, unsigned &Offset,
                                    msgpack::ArrayDocNode Args) = 0;
};

class MetadataStreamerMsgPackV3 : public MetadataStreamer {
protected:
  std::unique_ptr<msgpack::Document> HSAMetadataDoc =
      std::make_unique<msgpack::Document>();

  void dump(StringRef HSAMetadataString) const;

  void verify(StringRef HSAMetadataString) const;

  Optional<StringRef> getAccessQualifier(StringRef AccQual) const;

  Optional<StringRef> getAddressSpaceQualifier(unsigned AddressSpace) const;

  StringRef getValueKind(Type *Ty, StringRef TypeQual,
                         StringRef BaseTypeName) const;

  std::string getTypeName(Type *Ty, bool Signed) const;

  msgpack::ArrayDocNode getWorkGroupDimensions(MDNode *Node) const;

  msgpack::MapDocNode getHSAKernelProps(const MachineFunction &MF,
                                        const SIProgramInfo &ProgramInfo) const;

  void emitVersion() override;

  void emitPrintf(const Module &Mod);

  void emitKernelLanguage(const Function &Func, msgpack::MapDocNode Kern);

  void emitKernelAttrs(const Function &Func, msgpack::MapDocNode Kern);

  void emitKernelArgs(const MachineFunction &MF, msgpack::MapDocNode Kern);

  void emitKernelArg(const Argument &Arg, unsigned &Offset,
                     msgpack::ArrayDocNode Args);

  void emitKernelArg(const DataLayout &DL, Type *Ty, Align Alignment,
                     StringRef ValueKind, unsigned &Offset,
                     msgpack::ArrayDocNode Args, MaybeAlign PointeeAlign = None,
                     StringRef Name = "", StringRef TypeName = "",
                     StringRef BaseTypeName = "", StringRef AccQual = "",
                     StringRef TypeQual = "");

  void emitHiddenKernelArgs(const MachineFunction &MF, unsigned &Offset,
                            msgpack::ArrayDocNode Args) override;

  msgpack::DocNode &getRootMetadata(StringRef Key) {
    return HSAMetadataDoc->getRoot().getMap(/*Convert=*/true)[Key];
  }

  msgpack::DocNode &getHSAMetadataRoot() {
    return HSAMetadataDoc->getRoot();
  }

public:
  MetadataStreamerMsgPackV3() = default;
  ~MetadataStreamerMsgPackV3() = default;

  bool emitTo(AMDGPUTargetStreamer &TargetStreamer) override;

  void begin(const Module &Mod,
             const IsaInfo::AMDGPUTargetID &TargetID) override;

  void end() override;

  void emitKernel(const MachineFunction &MF,
                  const SIProgramInfo &ProgramInfo) override;
};

class MetadataStreamerMsgPackV4 : public MetadataStreamerMsgPackV3 {
protected:
  void emitVersion() override;
  void emitTargetID(const IsaInfo::AMDGPUTargetID &TargetID);

public:
  MetadataStreamerMsgPackV4() = default;
  ~MetadataStreamerMsgPackV4() = default;

  void begin(const Module &Mod,
             const IsaInfo::AMDGPUTargetID &TargetID) override;
};

class MetadataStreamerMsgPackV5 final : public MetadataStreamerMsgPackV4 {
protected:
  void emitVersion() override;
  void emitHiddenKernelArgs(const MachineFunction &MF, unsigned &Offset,
                            msgpack::ArrayDocNode Args) override;

public:
  MetadataStreamerMsgPackV5() = default;
  ~MetadataStreamerMsgPackV5() = default;
};

// TODO: Rename MetadataStreamerV2 -> MetadataStreamerYamlV2.
class MetadataStreamerYamlV2 final : public MetadataStreamer {
private:
  Metadata HSAMetadata;

  void dump(StringRef HSAMetadataString) const;

  void verify(StringRef HSAMetadataString) const;

  AccessQualifier getAccessQualifier(StringRef AccQual) const;

  AddressSpaceQualifier getAddressSpaceQualifier(unsigned AddressSpace) const;

  ValueKind getValueKind(Type *Ty, StringRef TypeQual,
                         StringRef BaseTypeName) const;

  std::string getTypeName(Type *Ty, bool Signed) const;

  std::vector<uint32_t> getWorkGroupDimensions(MDNode *Node) const;

  Kernel::CodeProps::Metadata getHSACodeProps(
      const MachineFunction &MF,
      const SIProgramInfo &ProgramInfo) const;
  Kernel::DebugProps::Metadata getHSADebugProps(
      const MachineFunction &MF,
      const SIProgramInfo &ProgramInfo) const;

  void emitPrintf(const Module &Mod);

  void emitKernelLanguage(const Function &Func);

  void emitKernelAttrs(const Function &Func);

  void emitKernelArgs(const Function &Func, const GCNSubtarget &ST);

  void emitKernelArg(const Argument &Arg);

  void emitKernelArg(const DataLayout &DL, Type *Ty, Align Alignment,
                     ValueKind ValueKind, MaybeAlign PointeeAlign = None,
                     StringRef Name = "", StringRef TypeName = "",
                     StringRef BaseTypeName = "", StringRef AccQual = "",
                     StringRef TypeQual = "");

  void emitHiddenKernelArgs(const Function &Func, const GCNSubtarget &ST);

  const Metadata &getHSAMetadata() const {
    return HSAMetadata;
  }

protected:
  void emitVersion() override;
  void emitHiddenKernelArgs(const MachineFunction &MF, unsigned &Offset,
                            msgpack::ArrayDocNode Args) override {
    llvm_unreachable("Dummy override should not be invoked!");
  }

public:
  MetadataStreamerYamlV2() = default;
  ~MetadataStreamerYamlV2() = default;

  bool emitTo(AMDGPUTargetStreamer &TargetStreamer) override;

  void begin(const Module &Mod,
             const IsaInfo::AMDGPUTargetID &TargetID) override;

  void end() override;

  void emitKernel(const MachineFunction &MF,
                  const SIProgramInfo &ProgramInfo) override;
};

} // end namespace HSAMD
} // end namespace AMDGPU
} // end namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_MCTARGETDESC_AMDGPUHSAMETADATASTREAMER_H
