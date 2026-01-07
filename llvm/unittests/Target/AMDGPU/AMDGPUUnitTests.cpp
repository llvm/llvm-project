//===--------- llvm/unittests/Target/AMDGPU/AMDGPUUnitTests.cpp -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AMDGPUUnitTests.h"
#include "AMDGPUTargetMachine.h"
#include "GCNSubtarget.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/TargetParser/TargetParser.h"
#include "gtest/gtest.h"

#include "AMDGPUGenSubtargetInfo.inc"

using namespace llvm;

std::once_flag flag;

void InitializeAMDGPUTarget() {
  std::call_once(flag, []() {
    LLVMInitializeAMDGPUTargetInfo();
    LLVMInitializeAMDGPUTarget();
    LLVMInitializeAMDGPUTargetMC();
  });
}

std::unique_ptr<const GCNTargetMachine>
llvm::createAMDGPUTargetMachine(std::string TStr, StringRef CPU, StringRef FS) {
  InitializeAMDGPUTarget();

  Triple TT(TStr);
  std::string Error;
  const Target *T = TargetRegistry::lookupTarget(TT, Error);
  if (!T)
    return nullptr;

  TargetOptions Options;
  return std::unique_ptr<GCNTargetMachine>(
      static_cast<GCNTargetMachine *>(T->createTargetMachine(
          TT, CPU, FS, Options, std::nullopt, std::nullopt)));
}

static cl::opt<bool> PrintCpuRegLimits(
    "print-cpu-reg-limits", cl::NotHidden, cl::init(false),
    cl::desc("force printing per AMDGPU CPU register limits"));

static bool checkMinMax(std::stringstream &OS, unsigned Occ, unsigned MinOcc,
                        unsigned MaxOcc,
                        std::function<unsigned(unsigned)> GetOcc,
                        std::function<unsigned(unsigned)> GetMinGPRs,
                        std::function<unsigned(unsigned)> GetMaxGPRs) {
  bool MinValid = true, MaxValid = true, RangeValid = true;
  unsigned MinGPRs = GetMinGPRs(Occ);
  unsigned MaxGPRs = GetMaxGPRs(Occ);
  unsigned RealOcc;

  if (MinGPRs >= MaxGPRs)
    RangeValid = false;
  else {
    RealOcc = GetOcc(MinGPRs);
    for (unsigned NumRegs = MinGPRs + 1; NumRegs <= MaxGPRs; ++NumRegs) {
      if (RealOcc != GetOcc(NumRegs)) {
        RangeValid = false;
        break;
      }
    }
  }

  if (RangeValid && RealOcc > MinOcc && RealOcc <= MaxOcc) {
    if (MinGPRs > 0 && GetOcc(MinGPRs - 1) <= RealOcc)
      MinValid = false;

    if (GetOcc(MaxGPRs + 1) >= RealOcc)
      MaxValid = false;
  }

  std::stringstream MinStr;
  MinStr << (MinValid ? ' ' : '<') << ' ' << std::setw(3) << MinGPRs << " (O"
         << GetOcc(MinGPRs) << ") " << (RangeValid ? ' ' : 'R');

  OS << std::left << std::setw(15) << MinStr.str() << std::setw(3) << MaxGPRs
     << " (O" << GetOcc(MaxGPRs) << ')' << (MaxValid ? "" : " >");

  return MinValid && MaxValid && RangeValid;
}

static const std::pair<StringRef, StringRef>
  EmptyFS = {"", ""},
  W32FS = {"+wavefrontsize32", "w32"},
  W64FS = {"+wavefrontsize64", "w64"};

using TestFuncTy = function_ref<bool(std::stringstream &, unsigned,
                                     const GCNSubtarget &, bool)>;

static bool testAndRecord(std::stringstream &Table, const GCNSubtarget &ST,
                          TestFuncTy test, unsigned DynamicVGPRBlockSize) {
  bool Success = true;
  unsigned MaxOcc = ST.getMaxWavesPerEU();
  for (unsigned Occ = MaxOcc; Occ > 0; --Occ) {
    Table << std::right << std::setw(3) << Occ << "    ";
    Success = test(Table, Occ, ST, DynamicVGPRBlockSize) && Success;
    Table << '\n';
  }
  return Success;
}

static void testGPRLimits(const char *RegName, bool TestW32W64,
                          TestFuncTy test) {
  SmallVector<StringRef> CPUs;
  AMDGPU::fillValidArchListAMDGCN(CPUs);

  std::map<std::string, SmallVector<std::string>> TablePerCPUs;
  for (auto CPUName : CPUs) {
    auto CanonCPUName =
        AMDGPU::getArchNameAMDGCN(AMDGPU::parseArchAMDGCN(CPUName));

    auto *FS = &EmptyFS;
    while (true) {
      auto TM = createAMDGPUTargetMachine("amdgcn-amd-", CPUName, FS->first);
      if (!TM)
        break;

      GCNSubtarget ST(TM->getTargetTriple(), std::string(TM->getTargetCPU()),
                      std::string(TM->getTargetFeatureString()), *TM);

      if (TestW32W64 &&
          ST.getFeatureBits().test(AMDGPU::FeatureWavefrontSize32))
        FS = &W32FS;

      std::stringstream Table;
      bool Success = testAndRecord(Table, ST, test, /*DynamicVGPRBlockSize=*/0);
      if (!Success || PrintCpuRegLimits)
        TablePerCPUs[Table.str()].push_back((CanonCPUName + FS->second).str());

      if (FS != &W32FS)
        break;

      FS = &W64FS;
    }
  }
  std::stringstream OS;
  for (auto &P : TablePerCPUs) {
    for (auto &CPUName : P.second)
      OS << ' ' << CPUName;
    OS << ":\nOcc    Min" << RegName << "        Max" << RegName << '\n'
       << P.first << '\n';
  }
  auto ErrStr = OS.str();
  EXPECT_TRUE(ErrStr.empty()) << ErrStr;
}

static void testDynamicVGPRLimits(StringRef CPUName, StringRef FS,
                                  TestFuncTy test) {
  auto TM = createAMDGPUTargetMachine("amdgcn-amd-", CPUName, FS);
  ASSERT_TRUE(TM) << "No target machine";

  GCNSubtarget ST(TM->getTargetTriple(), std::string(TM->getTargetCPU()),
                  std::string(TM->getTargetFeatureString()), *TM);

  auto testWithBlockSize = [&](unsigned DynamicVGPRBlockSize) {
    std::stringstream Table;
    bool Success = testAndRecord(Table, ST, test, DynamicVGPRBlockSize);
    EXPECT_TRUE(Success && !PrintCpuRegLimits)
        << CPUName << " dynamic VGPR block size " << DynamicVGPRBlockSize
        << ":\nOcc    MinVGPR        MaxVGPR\n"
        << Table.str() << '\n';
  };

  testWithBlockSize(16);
  testWithBlockSize(32);
}

TEST(AMDGPU, TestVGPRLimitsPerOccupancy) {
  auto test = [](std::stringstream &OS, unsigned Occ, const GCNSubtarget &ST,
                 unsigned DynamicVGPRBlockSize) {
    unsigned MaxVGPRNum = ST.getAddressableNumVGPRs(DynamicVGPRBlockSize);
    return checkMinMax(
        OS, Occ, ST.getOccupancyWithNumVGPRs(MaxVGPRNum, DynamicVGPRBlockSize),
        ST.getMaxWavesPerEU(),
        [&](unsigned NumGPRs) {
          return ST.getOccupancyWithNumVGPRs(NumGPRs, DynamicVGPRBlockSize);
        },
        [&](unsigned Occ) {
          return ST.getMinNumVGPRs(Occ, DynamicVGPRBlockSize);
        },
        [&](unsigned Occ) {
          return ST.getMaxNumVGPRs(Occ, DynamicVGPRBlockSize);
        });
  };

  testGPRLimits("VGPR", true, test);

  testDynamicVGPRLimits("gfx1200", "+wavefrontsize32", test);
}

static void testAbsoluteLimits(StringRef CPUName, StringRef FS,
                               unsigned DynamicVGPRBlockSize,
                               unsigned ExpectedMinOcc, unsigned ExpectedMaxOcc,
                               unsigned ExpectedMaxVGPRs) {
  auto TM = createAMDGPUTargetMachine("amdgcn-amd-", CPUName, FS);
  ASSERT_TRUE(TM) << "No target machine";

  GCNSubtarget ST(TM->getTargetTriple(), std::string(TM->getTargetCPU()),
                  std::string(TM->getTargetFeatureString()), *TM);

  // Test function without attributes.
  LLVMContext Context;
  Module M("", Context);
  Function *Func =
      Function::Create(FunctionType::get(Type::getVoidTy(Context), false),
                       GlobalValue::ExternalLinkage, "testFunc", &M);
  Func->setCallingConv(CallingConv::AMDGPU_CS_Chain);
  Func->addFnAttr("amdgpu-flat-work-group-size", "1,32");

  std::string DVGPRBlockSize = std::to_string(DynamicVGPRBlockSize);
  if (DynamicVGPRBlockSize)
    Func->addFnAttr("amdgpu-dynamic-vgpr-block-size", DVGPRBlockSize);

  auto Range = ST.getWavesPerEU(*Func);
  EXPECT_EQ(ExpectedMinOcc, Range.first) << CPUName << ' ' << FS;
  EXPECT_EQ(ExpectedMaxOcc, Range.second) << CPUName << ' ' << FS;
  EXPECT_EQ(ExpectedMaxVGPRs, ST.getMaxNumVGPRs(*Func)) << CPUName << ' ' << FS;
  EXPECT_EQ(ExpectedMaxVGPRs, ST.getAddressableNumVGPRs(DynamicVGPRBlockSize))
      << CPUName << ' ' << FS;

  // Function with requested 'amdgpu-waves-per-eu' in a valid range.
  Func->addFnAttr("amdgpu-waves-per-eu", "10,12");
  Range = ST.getWavesPerEU(*Func);
  EXPECT_EQ(10u, Range.first) << CPUName << ' ' << FS;
  EXPECT_EQ(12u, Range.second) << CPUName << ' ' << FS;
}

TEST(AMDGPU, TestOccupancyAbsoluteLimits) {
  // CPUName, Features, DynamicVGPRBlockSize; Expected MinOcc, MaxOcc, MaxVGPRs
  testAbsoluteLimits("gfx1200", "+wavefrontsize32", 0, 1, 16, 256);
  testAbsoluteLimits("gfx1200", "+wavefrontsize32", 16, 1, 16, 128);
  testAbsoluteLimits("gfx1200", "+wavefrontsize32", 32, 1, 16, 256);
}

static const char *printSubReg(const TargetRegisterInfo &TRI, unsigned SubReg) {
  return SubReg ? TRI.getSubRegIndexName(SubReg) : "<none>";
}

TEST(AMDGPU, TestReverseComposeSubRegIndices) {
  auto TM = createAMDGPUTargetMachine("amdgcn-amd-", "gfx900", "");
  if (!TM)
    return;
  GCNSubtarget ST(TM->getTargetTriple(), std::string(TM->getTargetCPU()),
                  std::string(TM->getTargetFeatureString()), *TM);

  const SIRegisterInfo *TRI = ST.getRegisterInfo();

#define EXPECT_SUBREG_EQ(A, B, Expect)                                         \
  do {                                                                         \
    unsigned Reversed = TRI->reverseComposeSubRegIndices(A, B);                \
    EXPECT_EQ(Reversed, Expect)                                                \
        << printSubReg(*TRI, A) << ", " << printSubReg(*TRI, B) << " => "      \
        << printSubReg(*TRI, Reversed) << ", *" << printSubReg(*TRI, Expect);  \
  } while (0);

  EXPECT_SUBREG_EQ(AMDGPU::NoSubRegister, AMDGPU::sub0, AMDGPU::sub0);
  EXPECT_SUBREG_EQ(AMDGPU::sub0, AMDGPU::NoSubRegister, AMDGPU::sub0);

  EXPECT_SUBREG_EQ(AMDGPU::sub0, AMDGPU::sub0, AMDGPU::sub0);

  EXPECT_SUBREG_EQ(AMDGPU::sub0, AMDGPU::sub1, AMDGPU::sub1);
  EXPECT_SUBREG_EQ(AMDGPU::sub1, AMDGPU::sub0, AMDGPU::NoSubRegister);

  EXPECT_SUBREG_EQ(AMDGPU::sub0_sub1, AMDGPU::sub0, AMDGPU::sub0);
  EXPECT_SUBREG_EQ(AMDGPU::sub0, AMDGPU::sub0_sub1, AMDGPU::sub0_sub1);

  EXPECT_SUBREG_EQ(AMDGPU::sub0_sub1_sub2_sub3, AMDGPU::sub0_sub1,
                   AMDGPU::sub0_sub1);
  EXPECT_SUBREG_EQ(AMDGPU::sub0_sub1, AMDGPU::sub0_sub1_sub2_sub3,
                   AMDGPU::sub0_sub1_sub2_sub3);

  EXPECT_SUBREG_EQ(AMDGPU::sub0_sub1_sub2_sub3, AMDGPU::sub1_sub2,
                   AMDGPU::sub1_sub2);
  EXPECT_SUBREG_EQ(AMDGPU::sub1_sub2, AMDGPU::sub0_sub1_sub2_sub3,
                   AMDGPU::NoSubRegister);

  EXPECT_SUBREG_EQ(AMDGPU::sub1_sub2_sub3, AMDGPU::sub0_sub1_sub2_sub3,
                   AMDGPU::NoSubRegister);
  EXPECT_SUBREG_EQ(AMDGPU::sub0_sub1_sub2_sub3, AMDGPU::sub1_sub2_sub3,
                   AMDGPU::sub1_sub2_sub3);

  EXPECT_SUBREG_EQ(AMDGPU::sub0, AMDGPU::sub30, AMDGPU::NoSubRegister);
  EXPECT_SUBREG_EQ(AMDGPU::sub30, AMDGPU::sub0, AMDGPU::NoSubRegister);

  EXPECT_SUBREG_EQ(AMDGPU::sub0, AMDGPU::sub31, AMDGPU::NoSubRegister);
  EXPECT_SUBREG_EQ(AMDGPU::sub31, AMDGPU::sub0, AMDGPU::NoSubRegister);

  EXPECT_SUBREG_EQ(AMDGPU::sub0_sub1, AMDGPU::sub30, AMDGPU::NoSubRegister);
  EXPECT_SUBREG_EQ(AMDGPU::sub30, AMDGPU::sub0_sub1, AMDGPU::NoSubRegister);

  EXPECT_SUBREG_EQ(AMDGPU::sub0_sub1, AMDGPU::sub30_sub31,
                   AMDGPU::NoSubRegister);
  EXPECT_SUBREG_EQ(AMDGPU::sub30_sub31, AMDGPU::sub0_sub1,
                   AMDGPU::NoSubRegister);

  for (unsigned SubIdx0 = 1, LastSubReg = TRI->getNumSubRegIndices();
       SubIdx0 != LastSubReg; ++SubIdx0) {
    for (unsigned SubIdx1 = 1; SubIdx1 != LastSubReg; ++SubIdx1) {
      if (unsigned ForwardCompose =
              TRI->composeSubRegIndices(SubIdx0, SubIdx1)) {
        unsigned ReverseComposed =
            TRI->reverseComposeSubRegIndices(SubIdx0, ForwardCompose);
        EXPECT_EQ(ReverseComposed, SubIdx1);
      }

      if (unsigned ReverseCompose =
              TRI->reverseComposeSubRegIndices(SubIdx0, SubIdx1)) {
        unsigned Recompose = TRI->composeSubRegIndices(SubIdx0, ReverseCompose);
        EXPECT_EQ(Recompose, SubIdx1);
      }
    }
  }
}

TEST(AMDGPU, TestGetNamedOperandIdx) {
  std::unique_ptr<const GCNTargetMachine> TM =
      createAMDGPUTargetMachine("amdgcn-amd-", "gfx900", "");
  if (!TM)
    return;
  const MCInstrInfo *MCII = TM->getMCInstrInfo();

  for (unsigned Opcode = 0, E = MCII->getNumOpcodes(); Opcode != E; ++Opcode) {
    const MCInstrDesc &Desc = MCII->get(Opcode);
    for (unsigned Idx = 0; Idx < Desc.getNumOperands(); ++Idx) {
      AMDGPU::OpName OpName = AMDGPU::getOperandIdxName(Opcode, Idx);
      if (OpName == AMDGPU::OpName::NUM_OPERAND_NAMES)
        continue;
      int16_t RetrievedIdx = AMDGPU::getNamedOperandIdx(Opcode, OpName);
      EXPECT_EQ(Idx, static_cast<unsigned>(RetrievedIdx))
          << "Opcode " << Opcode << " (" << MCII->getName(Opcode) << ')';
    }
  }
}
