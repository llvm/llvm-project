//===-- AMDGPUAsmPrinter.h - Print AMDGPU assembly code ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
/// \brief AMDGPU Assembly printer class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPUASMPRINTER_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPUASMPRINTER_H

#include "AMDGPU.h"
#include "AMDKernelCodeT.h"
#include "MCTargetDesc/AMDGPUHSAMetadataStreamer.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace llvm {

class AMDGPUTargetStreamer;
class MCOperand;
class SISubtarget;

class AMDGPUAsmPrinter final : public AsmPrinter {
private:
  // Track resource usage for callee functions.
  struct SIFunctionResourceInfo {
    // Track the number of explicitly used VGPRs. Special registers reserved at
    // the end are tracked separately.
    int32_t NumVGPR = 0;
    int32_t NumExplicitSGPR = 0;
    uint64_t PrivateSegmentSize = 0;
    bool UsesVCC = false;
    bool UsesFlatScratch = false;
    bool HasDynamicallySizedStack = false;
    bool HasRecursion = false;

    int32_t getTotalNumSGPRs(const SISubtarget &ST) const;
  };

  // Track resource usage for kernels / entry functions.
  struct SIProgramInfo {
    // Fields set in PGM_RSRC1 pm4 packet.
    uint32_t VGPRBlocks = 0;
    uint32_t SGPRBlocks = 0;
    uint32_t Priority = 0;
    uint32_t FloatMode = 0;
    uint32_t Priv = 0;
    uint32_t DX10Clamp = 0;
    uint32_t DebugMode = 0;
    uint32_t IEEEMode = 0;
    uint64_t ScratchSize = 0;

    uint64_t ComputePGMRSrc1 = 0;

    // Fields set in PGM_RSRC2 pm4 packet.
    uint32_t LDSBlocks = 0;
    uint32_t ScratchBlocks = 0;

    uint64_t ComputePGMRSrc2 = 0;

    uint32_t NumVGPR = 0;
    uint32_t NumSGPR = 0;
    uint32_t LDSSize = 0;
    bool FlatUsed = false;

    // Number of SGPRs that meets number of waves per execution unit request.
    uint32_t NumSGPRsForWavesPerEU = 0;

    // Number of VGPRs that meets number of waves per execution unit request.
    uint32_t NumVGPRsForWavesPerEU = 0;

    // If ReservedVGPRCount is 0 then must be 0. Otherwise, this is the first
    // fixed VGPR number reserved.
    uint16_t ReservedVGPRFirst = 0;

    // The number of consecutive VGPRs reserved.
    uint16_t ReservedVGPRCount = 0;

    // Fixed SGPR number used to hold wave scratch offset for entire kernel
    // execution, or std::numeric_limits<uint16_t>::max() if the register is not
    // used or not known.
    uint16_t DebuggerWavefrontPrivateSegmentOffsetSGPR =
        std::numeric_limits<uint16_t>::max();

    // Fixed SGPR number of the first 4 SGPRs used to hold scratch V# for entire
    // kernel execution, or std::numeric_limits<uint16_t>::max() if the register
    // is not used or not known.
    uint16_t DebuggerPrivateSegmentBufferSGPR =
        std::numeric_limits<uint16_t>::max();

    // Whether there is recursion, dynamic allocas, indirect calls or some other
    // reason there may be statically unknown stack usage.
    bool DynamicCallStack = false;

    // Bonus information for debugging.
    bool VCCUsed = false;

    SIProgramInfo() = default;
  };

  SIProgramInfo CurrentProgramInfo;
  DenseMap<const Function *, SIFunctionResourceInfo> CallGraphResourceInfo;

  AMDGPU::HSAMD::MetadataStreamer HSAMetadataStream;
  std::map<uint32_t, uint32_t> PALMetadataMap;

  uint64_t getFunctionCodeSize(const MachineFunction &MF) const;
  SIFunctionResourceInfo analyzeResourceUsage(const MachineFunction &MF) const;

  void readPALMetadata(Module &M);
  void getSIProgramInfo(SIProgramInfo &Out, const MachineFunction &MF);
  void getAmdKernelCode(amd_kernel_code_t &Out, const SIProgramInfo &KernelInfo,
                        const MachineFunction &MF) const;
  void findNumUsedRegistersSI(const MachineFunction &MF,
                              unsigned &NumSGPR,
                              unsigned &NumVGPR) const;

  AMDGPU::HSAMD::Kernel::CodeProps::Metadata getHSACodeProps(
      const MachineFunction &MF,
      const SIProgramInfo &ProgramInfo) const;
  AMDGPU::HSAMD::Kernel::DebugProps::Metadata getHSADebugProps(
      const MachineFunction &MF,
      const SIProgramInfo &ProgramInfo) const;

  /// \brief Emit register usage information so that the GPU driver
  /// can correctly setup the GPU state.
  void EmitProgramInfoR600(const MachineFunction &MF);
  void EmitProgramInfoSI(const MachineFunction &MF,
                         const SIProgramInfo &KernelInfo);
  void EmitPALMetadata(const MachineFunction &MF,
                       const SIProgramInfo &KernelInfo);
  void emitCommonFunctionComments(uint32_t NumVGPR,
                                  uint32_t NumSGPR,
                                  uint64_t ScratchSize,
                                  uint64_t CodeSize);

public:
  explicit AMDGPUAsmPrinter(TargetMachine &TM,
                            std::unique_ptr<MCStreamer> Streamer);

  StringRef getPassName() const override;

  const MCSubtargetInfo* getSTI() const;

  AMDGPUTargetStreamer* getTargetStreamer() const;

  bool doFinalization(Module &M) override;
  bool runOnMachineFunction(MachineFunction &MF) override;

  /// \brief Wrapper for MCInstLowering.lowerOperand() for the tblgen'erated
  /// pseudo lowering.
  bool lowerOperand(const MachineOperand &MO, MCOperand &MCOp) const;

  /// \brief Lower the specified LLVM Constant to an MCExpr.
  /// The AsmPrinter::lowerConstantof does not know how to lower
  /// addrspacecast, therefore they should be lowered by this function.
  const MCExpr *lowerConstant(const Constant *CV) override;

  /// \brief tblgen'erated driver function for lowering simple MI->MC pseudo
  /// instructions.
  bool emitPseudoExpansionLowering(MCStreamer &OutStreamer,
                                   const MachineInstr *MI);

  /// Implemented in AMDGPUMCInstLower.cpp
  void EmitInstruction(const MachineInstr *MI) override;

  void EmitFunctionBodyStart() override;

  void EmitFunctionEntryLabel() override;

  void EmitBasicBlockStart(const MachineBasicBlock &MBB) const override;

  void EmitGlobalVariable(const GlobalVariable *GV) override;

  void EmitStartOfAsmFile(Module &M) override;

  void EmitEndOfAsmFile(Module &M) override;

  bool isBlockOnlyReachableByFallthrough(
    const MachineBasicBlock *MBB) const override;

  bool PrintAsmOperand(const MachineInstr *MI, unsigned OpNo,
                       unsigned AsmVariant, const char *ExtraCode,
                       raw_ostream &O) override;

protected:
  mutable std::vector<std::string> DisasmLines, HexLines;
  mutable size_t DisasmLineMaxLen;
  AMDGPUAS AMDGPUASI;
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_AMDGPUASMPRINTER_H
