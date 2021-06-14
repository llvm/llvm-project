//===-- AMDGPUAsmPrinter.h - Print AMDGPU assembly code ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// AMDGPU Assembly printer class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPUASMPRINTER_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPUASMPRINTER_H

#include "SIProgramInfo.h"
#include "llvm/CodeGen/AsmPrinter.h"

struct amd_kernel_code_t;

namespace llvm {

class AMDGPUMachineFunction;
class AMDGPUTargetStreamer;
class MCCodeEmitter;
class MCOperand;
class GCNSubtarget;

namespace AMDGPU {
namespace HSAMD {
class MetadataStreamer;
}
} // namespace AMDGPU

namespace amdhsa {
struct kernel_descriptor_t;
}

class AMDGPUAsmPrinter final : public AsmPrinter {
private:
  // Track resource usage for callee functions.
  struct SIFunctionResourceInfo {
    // Track the number of explicitly used VGPRs. Special registers reserved at
    // the end are tracked separately.
    int32_t NumVGPR = 0;
    int32_t NumAGPR = 0;
    int32_t NumExplicitSGPR = 0;
    uint64_t PrivateSegmentSize = 0;
    bool UsesVCC = false;
    bool UsesFlatScratch = false;
    bool HasDynamicallySizedStack = false;
    bool HasRecursion = false;

    int32_t getTotalNumSGPRs(const GCNSubtarget &ST) const;
    int32_t getTotalNumVGPRs(const GCNSubtarget &ST) const;
  };

  void initializeTargetID(const Module &M);

  bool doInitialization(Module &M) override;

  SIProgramInfo CurrentProgramInfo;
  DenseMap<const Function *, SIFunctionResourceInfo> CallGraphResourceInfo;

  std::unique_ptr<AMDGPU::HSAMD::MetadataStreamer> HSAMetadataStream;

  MCCodeEmitter *DumpCodeInstEmitter = nullptr;

  uint64_t getFunctionCodeSize(const MachineFunction &MF) const;
  SIFunctionResourceInfo analyzeResourceUsage(const MachineFunction &MF) const;

  void getSIProgramInfo(SIProgramInfo &Out, const MachineFunction &MF);
  void getAmdKernelCode(amd_kernel_code_t &Out, const SIProgramInfo &KernelInfo,
                        const MachineFunction &MF) const;
  void findNumUsedRegistersSI(const MachineFunction &MF,
                              unsigned &NumSGPR,
                              unsigned &NumVGPR) const;

  /// Emit register usage information so that the GPU driver
  /// can correctly setup the GPU state.
  void EmitProgramInfoSI(const MachineFunction &MF,
                         const SIProgramInfo &KernelInfo);
  void EmitPALMetadata(const MachineFunction &MF,
                       const SIProgramInfo &KernelInfo);
  void emitPALFunctionMetadata(const MachineFunction &MF);
  void emitCommonFunctionComments(uint32_t NumVGPR,
                                  Optional<uint32_t> NumAGPR,
                                  uint32_t TotalNumVGPR,
                                  uint32_t NumSGPR,
                                  uint64_t ScratchSize,
                                  uint64_t CodeSize,
                                  const AMDGPUMachineFunction* MFI);

  uint16_t getAmdhsaKernelCodeProperties(
      const MachineFunction &MF) const;

  amdhsa::kernel_descriptor_t getAmdhsaKernelDescriptor(
      const MachineFunction &MF,
      const SIProgramInfo &PI) const;

public:
  explicit AMDGPUAsmPrinter(TargetMachine &TM,
                            std::unique_ptr<MCStreamer> Streamer);

  // To memoize max SGPR usage of non-kernel functions of the module.
  unsigned NonKernelMaxSGPRs = 0;
  // To memoize max VGPR usage of non-kernel functions of the module.
  unsigned NonKernelMaxVGPRs = 0;

  StringRef getPassName() const override;

  const MCSubtargetInfo* getGlobalSTI() const;

  AMDGPUTargetStreamer* getTargetStreamer() const;

  bool doFinalization(Module &M) override;
  bool runOnMachineFunction(MachineFunction &MF) override;

  /// Wrapper for MCInstLowering.lowerOperand() for the tblgen'erated
  /// pseudo lowering.
  bool lowerOperand(const MachineOperand &MO, MCOperand &MCOp) const;

  /// Lower the specified LLVM Constant to an MCExpr.
  /// The AsmPrinter::lowerConstantof does not know how to lower
  /// addrspacecast, therefore they should be lowered by this function.
  const MCExpr *lowerConstant(const Constant *CV) override;

  /// tblgen'erated driver function for lowering simple MI->MC pseudo
  /// instructions.
  bool emitPseudoExpansionLowering(MCStreamer &OutStreamer,
                                   const MachineInstr *MI);

  /// Implemented in AMDGPUMCInstLower.cpp
  void emitInstruction(const MachineInstr *MI) override;

  void emitFunctionBodyStart() override;

  void emitFunctionBodyEnd() override;

  void emitFunctionEntryLabel() override;

  void emitBasicBlockStart(const MachineBasicBlock &MBB) override;

  void emitGlobalVariable(const GlobalVariable *GV) override;

  void emitStartOfAsmFile(Module &M) override;

  void emitEndOfAsmFile(Module &M) override;

  bool isBlockOnlyReachableByFallthrough(
    const MachineBasicBlock *MBB) const override;

  bool PrintAsmOperand(const MachineInstr *MI, unsigned OpNo,
                       const char *ExtraCode, raw_ostream &O) override;

protected:
  std::vector<std::string> DisasmLines, HexLines;
  size_t DisasmLineMaxLen;
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_AMDGPUASMPRINTER_H
