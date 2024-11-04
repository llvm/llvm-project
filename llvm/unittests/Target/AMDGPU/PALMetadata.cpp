//===- llvm/unittests/MC/AMDGPU/PALMetadata.cpp ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AMDGPUTargetMachine.h"
#include "GCNSubtarget.h"
#include "SIProgramInfo.h"
#include "Utils/AMDGPUPALMetadata.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MCTargetOptions.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "gtest/gtest.h"

using namespace llvm;

class PALMetadata : public testing::Test {
protected:
  std::unique_ptr<GCNTargetMachine> TM;
  std::unique_ptr<LLVMContext> Ctx;
  std::unique_ptr<GCNSubtarget> ST;
  std::unique_ptr<MachineModuleInfo> MMI;
  std::unique_ptr<MachineFunction> MF;
  std::unique_ptr<Module> M;
  AMDGPUPALMetadata MD;

  static void SetUpTestSuite() {
    LLVMInitializeAMDGPUTargetInfo();
    LLVMInitializeAMDGPUTarget();
    LLVMInitializeAMDGPUTargetMC();
  }

  PALMetadata() {
    StringRef Triple = "amdgcn--amdpal";
    StringRef CPU = "gfx1010";
    StringRef FS = "";

    std::string Error;
    const Target *TheTarget = TargetRegistry::lookupTarget(Triple, Error);
    TargetOptions Options;

    TM.reset(static_cast<GCNTargetMachine *>(TheTarget->createTargetMachine(
        Triple, CPU, FS, Options, std::nullopt, std::nullopt)));

    Ctx = std::make_unique<LLVMContext>();
    M = std::make_unique<Module>("Module", *Ctx);
    M->setDataLayout(TM->createDataLayout());
    auto *FType = FunctionType::get(Type::getVoidTy(*Ctx), false);
    auto *F = Function::Create(FType, GlobalValue::ExternalLinkage, "Test", *M);
    MMI = std::make_unique<MachineModuleInfo>(TM.get());

    ST = std::make_unique<GCNSubtarget>(TM->getTargetTriple(),
                                        TM->getTargetCPU(),
                                        TM->getTargetFeatureString(), *TM);

    MF = std::make_unique<MachineFunction>(*F, *TM, *ST, 1, *MMI);
  }
};

TEST_F(PALMetadata, ResourceRegisterSetORsResolvableUnknown) {
  StringRef yaml = "---\n"
                   "amdpal.pipelines:\n"
                   "  - .hardware_stages:\n"
                   "      .es:\n"
                   "        .entry_point:    Test\n"
                   "        .scratch_memory_size: 0\n"
                   "        .sgpr_count:     0x1\n"
                   "        .vgpr_count:     0x1\n"
                   "    .registers:\n"
                   "      \'0x2c4a (SPI_SHADER_PGM_RSRC1_VS)\': 0x2f0000\n"
                   "      \'0x2c4b (SPI_SHADER_PGM_RSRC2_VS)\': 0\n"
                   "...\n";

  MCContext &MCCtx = MF->getContext();
  auto CC = CallingConv::AMDGPU_VS;
  MD.setFromString(yaml);
  MD.setRsrc2(CC, MCConstantExpr::create(42, MCCtx), MCCtx);
  MCSymbol *Sym = MCCtx.getOrCreateSymbol("Unknown");
  MD.setRsrc2(CC, MCSymbolRefExpr::create(Sym, MCCtx), MCCtx);
  EXPECT_FALSE(MD.resolvedAllMCExpr());

  MD.setRsrc2(CC, MCConstantExpr::create(0xff00, MCCtx), MCCtx);
  Sym->setVariableValue(MCConstantExpr::create(0xffff0000, MCCtx));
  std::string Output;
  MD.toString(Output);

  EXPECT_TRUE(MD.resolvedAllMCExpr());

  auto n = Output.find("\'0x2c4b (SPI_SHADER_PGM_RSRC2_VS)\': 0xffffff2a");
  EXPECT_TRUE(n != std::string::npos);
}

TEST_F(PALMetadata, ResourceRegisterSetORsResolvableUnknowns) {
  StringRef yaml = "---\n"
                   "amdpal.pipelines:\n"
                   "  - .hardware_stages:\n"
                   "      .es:\n"
                   "        .entry_point:    Test\n"
                   "        .scratch_memory_size: 0\n"
                   "        .sgpr_count:     0x1\n"
                   "        .vgpr_count:     0x1\n"
                   "    .registers:\n"
                   "      \'0x2c4a (SPI_SHADER_PGM_RSRC1_VS)\': 0x2f0000\n"
                   "      \'0x2c4b (SPI_SHADER_PGM_RSRC2_VS)\': 0\n"
                   "...\n";

  MCContext &MCCtx = MF->getContext();
  auto CC = CallingConv::AMDGPU_VS;
  MD.setFromString(yaml);
  MCSymbol *SymOne = MCCtx.getOrCreateSymbol("UnknownOne");
  MD.setRsrc2(CC, MCSymbolRefExpr::create(SymOne, MCCtx), MCCtx);

  MD.setRsrc2(CC, MCConstantExpr::create(42, MCCtx), MCCtx);

  MCSymbol *SymTwo = MCCtx.getOrCreateSymbol("UnknownTwo");
  MD.setRsrc2(CC, MCSymbolRefExpr::create(SymTwo, MCCtx), MCCtx);
  EXPECT_FALSE(MD.resolvedAllMCExpr());

  SymOne->setVariableValue(MCConstantExpr::create(0xffff0000, MCCtx));
  SymTwo->setVariableValue(MCConstantExpr::create(0x0000ff00, MCCtx));

  std::string Output;
  MD.toString(Output);

  EXPECT_TRUE(MD.resolvedAllMCExpr());

  auto n = Output.find("\'0x2c4b (SPI_SHADER_PGM_RSRC2_VS)\': 0xffffff2a");
  EXPECT_TRUE(n != std::string::npos);
}

TEST_F(PALMetadata, ResourceRegisterSetORsPreset) {
  StringRef yaml = "---\n"
                   "amdpal.pipelines:\n"
                   "  - .hardware_stages:\n"
                   "      .es:\n"
                   "        .entry_point:    Test\n"
                   "        .scratch_memory_size: 0\n"
                   "        .sgpr_count:     0x1\n"
                   "        .vgpr_count:     0x1\n"
                   "    .registers:\n"
                   "      \'0x2c4a (SPI_SHADER_PGM_RSRC1_VS)\': 0x2f0000\n"
                   "      \'0x2c4b (SPI_SHADER_PGM_RSRC2_VS)\': 0x2a\n"
                   "...\n";

  MCContext &MCCtx = MF->getContext();
  auto CC = CallingConv::AMDGPU_VS;
  MD.setFromString(yaml);
  MCSymbol *Sym = MCCtx.getOrCreateSymbol("Unknown");
  MD.setRsrc2(CC, MCSymbolRefExpr::create(Sym, MCCtx), MCCtx);
  MD.setRsrc2(CC, MCConstantExpr::create(0xff00, MCCtx), MCCtx);
  Sym->setVariableValue(MCConstantExpr::create(0xffff0000, MCCtx));
  std::string Output;
  MD.toString(Output);

  auto n = Output.find("\'0x2c4b (SPI_SHADER_PGM_RSRC2_VS)\': 0xffffff2a");
  EXPECT_TRUE(n != std::string::npos);
}

TEST_F(PALMetadata, ResourceRegisterSetORs) {
  StringRef yaml = "---\n"
                   "amdpal.pipelines:\n"
                   "  - .hardware_stages:\n"
                   "      .es:\n"
                   "        .entry_point:    Test\n"
                   "        .scratch_memory_size: 0\n"
                   "        .sgpr_count:     0x1\n"
                   "        .vgpr_count:     0x1\n"
                   "    .registers:\n"
                   "      \'0x2c4a (SPI_SHADER_PGM_RSRC1_VS)\': 0x2f0000\n"
                   "      \'0x2c4b (SPI_SHADER_PGM_RSRC2_VS)\': 0\n"
                   "...\n";

  MCContext &MCCtx = MF->getContext();
  auto CC = CallingConv::AMDGPU_VS;
  MD.setFromString(yaml);
  MCSymbol *Sym = MCCtx.getOrCreateSymbol("Unknown");
  MD.setRsrc2(CC, MCSymbolRefExpr::create(Sym, MCCtx), MCCtx);
  MD.setRsrc2(CC, 42);
  MD.setRsrc2(CC, MCConstantExpr::create(0xff00, MCCtx), MCCtx);
  Sym->setVariableValue(MCConstantExpr::create(0xffff0000, MCCtx));
  std::string Output;
  MD.toString(Output);

  auto n = Output.find("\'0x2c4b (SPI_SHADER_PGM_RSRC2_VS)\': 0xffffff2a");
  EXPECT_TRUE(n != std::string::npos);
}

TEST_F(PALMetadata, ResourceRegisterSetUnresolvedSym) {
  StringRef yaml = "---\n"
                   "amdpal.pipelines:\n"
                   "  - .hardware_stages:\n"
                   "      .es:\n"
                   "        .entry_point:    Test\n"
                   "        .scratch_memory_size: 0\n"
                   "        .sgpr_count:     0x1\n"
                   "        .vgpr_count:     0x1\n"
                   "    .registers:\n"
                   "      \'0x2c4a (SPI_SHADER_PGM_RSRC1_VS)\': 0x2f0000\n"
                   "      \'0x2c4b (SPI_SHADER_PGM_RSRC2_VS)\': 0\n"
                   "...\n";

  MCContext &MCCtx = MF->getContext();
  auto CC = CallingConv::AMDGPU_VS;
  MD.setFromString(yaml);
  MCSymbol *Sym = MCCtx.getOrCreateSymbol("Unknown");
  MD.setRsrc2(CC, MCSymbolRefExpr::create(Sym, MCCtx), MCCtx);
  MD.setRsrc2(CC, MCConstantExpr::create(0xff00, MCCtx), MCCtx);
  std::string Output;

  MD.toString(Output);
  EXPECT_FALSE(MD.resolvedAllMCExpr());
}

TEST_F(PALMetadata, ResourceRegisterSetNoEmitUnresolved) {
  StringRef yaml = "---\n"
                   "amdpal.pipelines:\n"
                   "  - .hardware_stages:\n"
                   "      .es:\n"
                   "        .entry_point:    Test\n"
                   "        .scratch_memory_size: 0\n"
                   "        .sgpr_count:     0x1\n"
                   "        .vgpr_count:     0x1\n"
                   "    .registers:\n"
                   "      \'0x2c4a (SPI_SHADER_PGM_RSRC1_VS)\': 0x2f0000\n"
                   "      \'0x2c4b (SPI_SHADER_PGM_RSRC2_VS)\': 0\n"
                   "...\n";

  MCContext &MCCtx = MF->getContext();
  auto CC = CallingConv::AMDGPU_VS;
  MD.setFromString(yaml);
  MCSymbol *Sym = MCCtx.getOrCreateSymbol("Unknown");
  MD.setRsrc2(CC, MCSymbolRefExpr::create(Sym, MCCtx), MCCtx);
  MD.setRsrc2(CC, MCConstantExpr::create(0xff00, MCCtx), MCCtx);

  EXPECT_FALSE(MD.resolvedAllMCExpr());
}
