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

#ifndef LLVM_LIB_TARGET_R600_AMDGPUASMPRINTER_H
#define LLVM_LIB_TARGET_R600_AMDGPUASMPRINTER_H

#include "llvm/CodeGen/AsmPrinter.h"
#include <vector>

namespace llvm {

class AMDGPUAsmPrinter : public AsmPrinter {
private:
  struct SIProgramInfo {
    SIProgramInfo() :
      VGPRBlocks(0),
      SGPRBlocks(0),
      Priority(0),
      FloatMode(0),
      Priv(0),
      DX10Clamp(0),
      DebugMode(0),
      IEEEMode(0),
      ScratchSize(0),
      ComputePGMRSrc1(0),
      LDSBlocks(0),
      ScratchBlocks(0),
      ComputePGMRSrc2(0),
      NumVGPR(0),
      NumSGPR(0),
      FlatUsed(false),
      VCCUsed(false),
      CodeLen(0) {}

    // Fields set in PGM_RSRC1 pm4 packet.
    uint32_t VGPRBlocks;
    uint32_t SGPRBlocks;
    uint32_t Priority;
    uint32_t FloatMode;
    uint32_t Priv;
    uint32_t DX10Clamp;
    uint32_t DebugMode;
    uint32_t IEEEMode;
    uint32_t ScratchSize;

    uint64_t ComputePGMRSrc1;

    // Fields set in PGM_RSRC2 pm4 packet.
    uint32_t LDSBlocks;
    uint32_t ScratchBlocks;

    uint64_t ComputePGMRSrc2;

    uint32_t NumVGPR;
    uint32_t NumSGPR;
    uint32_t LDSSize;
    bool FlatUsed;

    // Bonus information for debugging.
    bool VCCUsed;
    uint64_t CodeLen;
  };

  void getSIProgramInfo(SIProgramInfo &Out, const MachineFunction &MF) const;
  void findNumUsedRegistersSI(const MachineFunction &MF,
                              unsigned &NumSGPR,
                              unsigned &NumVGPR) const;

  /// \brief Emit register usage information so that the GPU driver
  /// can correctly setup the GPU state.
  void EmitProgramInfoR600(const MachineFunction &MF);
  void EmitProgramInfoSI(const MachineFunction &MF, const SIProgramInfo &KernelInfo);
  void EmitAmdKernelCodeT(const MachineFunction &MF,
                          const SIProgramInfo &KernelInfo) const;

public:
  explicit AMDGPUAsmPrinter(TargetMachine &TM,
                            std::unique_ptr<MCStreamer> Streamer);

  bool runOnMachineFunction(MachineFunction &MF) override;

  const char *getPassName() const override {
    return "AMDGPU Assembly Printer";
  }

  /// Implemented in AMDGPUMCInstLower.cpp
  void EmitInstruction(const MachineInstr *MI) override;

  void EmitFunctionBodyStart() override;

  void EmitEndOfAsmFile(Module &M) override;

  bool PrintAsmOperand(const MachineInstr *MI, unsigned OpNo,
                       unsigned AsmVariant, const char *ExtraCode,
                       raw_ostream &O) override;

protected:
  std::vector<std::string> DisasmLines, HexLines;
  size_t DisasmLineMaxLen;
};

} // End anonymous llvm

#endif
