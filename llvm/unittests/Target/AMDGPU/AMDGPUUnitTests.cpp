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
#include "llvm/Support/TargetParser.h"
#include "llvm/Support/TargetSelect.h"
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

  std::string Error;
  const Target *T = TargetRegistry::lookupTarget(TStr, Error);
  if (!T)
    return nullptr;

  TargetOptions Options;
  return std::unique_ptr<GCNTargetMachine>(
      static_cast<GCNTargetMachine *>(T->createTargetMachine(
          TStr, CPU, FS, Options, std::nullopt, std::nullopt)));
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

static void testGPRLimits(
    const char *RegName, bool TestW32W64,
    std::function<bool(std::stringstream &, unsigned, GCNSubtarget &)> test) {
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
      bool Success = true;
      unsigned MaxOcc = ST.getMaxWavesPerEU();
      for (unsigned Occ = MaxOcc; Occ > 0; --Occ) {
        Table << std::right << std::setw(3) << Occ << "    ";
        Success = test(Table, Occ, ST) && Success;
        Table << '\n';
      }
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

TEST(AMDGPU, TestVGPRLimitsPerOccupancy) {
  testGPRLimits("VGPR", true, [](std::stringstream &OS, unsigned Occ,
                                 GCNSubtarget &ST) {
    unsigned MaxVGPRNum = ST.getAddressableNumVGPRs();
    return checkMinMax(
        OS, Occ, ST.getOccupancyWithNumVGPRs(MaxVGPRNum), ST.getMaxWavesPerEU(),
        [&](unsigned NumGPRs) { return ST.getOccupancyWithNumVGPRs(NumGPRs); },
        [&](unsigned Occ) { return ST.getMinNumVGPRs(Occ); },
        [&](unsigned Occ) { return ST.getMaxNumVGPRs(Occ); });
  });
}
