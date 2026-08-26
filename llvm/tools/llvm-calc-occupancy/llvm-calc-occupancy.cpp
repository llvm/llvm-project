//===-- llvm-calc-occupancy.cpp - AMDGPU occupancy calculator -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A small standalone utility that answers "what occupancy do I get?" for an
// AMDGPU kernel, given some subset of its resource usage: workgroup size,
// VGPRs, SGPRs and LDS. Fields that are left unspecified are treated as
// unconstrained, and the result is reported as a range (waves per EU).
//
// It reuses the compiler's own occupancy math (GCNSubtarget) so the numbers
// match what the backend would compute for the same inputs.
//
// TODO: This links the AMDGPU codegen libraries only because the occupancy
// math currently lives in GCNSubtarget. Once that subtarget information is
// exposed through TargetParser, this tool should depend on TargetParser alone
// and drop the codegen dependency.
//
// Example:
//   llvm-calc-occupancy -mcpu=gfx90a --wg-size=512 --vgprs=50 --sgprs=30 \
//                       --lds=103kb
//
//===----------------------------------------------------------------------===//

#include "AMDGPUTargetMachine.h"
#include "GCNSubtarget.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Triple.h"
#include <optional>

using namespace llvm;

namespace {
cl::OptionCategory OccCategory("llvm-calc-occupancy options");

cl::opt<std::string> TripleName("mtriple", cl::desc("Target triple"),
                                cl::init("amdgpu-amd-amdhsa"),
                                cl::cat(OccCategory));

cl::opt<std::string> MCPU("mcpu", cl::desc("Target GPU (e.g. gfx90a)"),
                          cl::init(""), cl::cat(OccCategory));

cl::opt<std::string> MAttr("mattr",
                           cl::desc("Comma-separated subtarget features "
                                    "(e.g. +wavefrontsize32)"),
                           cl::init(""), cl::cat(OccCategory));

cl::opt<std::string>
    WGSizeStr("wg-size",
              cl::desc("Flat workgroup size: single value 'N' or range "
                       "'MIN:MAX' (default: 1:1024)"),
              cl::init(""), cl::cat(OccCategory));
cl::alias WGSizeAlias("flat-workgroup-size", cl::aliasopt(WGSizeStr));

cl::opt<int> NumVGPRs("vgprs", cl::desc("VGPRs used per lane (default: none)"),
                      cl::init(-1), cl::cat(OccCategory));

cl::opt<int> NumSGPRs("sgprs", cl::desc("SGPRs used per wave (default: none)"),
                      cl::init(-1), cl::cat(OccCategory));

cl::opt<std::string> LDSStr("lds",
                            cl::desc("LDS bytes per workgroup, accepts k/kb/m "
                                     "suffixes (default: 0)"),
                            cl::init(""), cl::cat(OccCategory));

cl::opt<unsigned>
    DynVGPRBlockSize("dynamic-vgpr-block-size",
                     cl::desc("Dynamic VGPR block size (0 = disabled)"),
                     cl::init(0), cl::cat(OccCategory));

cl::opt<bool> ShowLimits("limits",
                         cl::desc("Print the per-occupancy VGPR/SGPR limit "
                                  "table for this GPU"),
                         cl::init(false), cl::cat(OccCategory));
} // namespace

// Parse a byte size with an optional binary suffix (k/kb/kib/m/mb/mib), all
// base 1024. A bare number is interpreted as bytes.
static bool parseSize(StringRef S, uint64_t &Out) {
  S = S.trim();
  if (S.empty())
    return false;
  uint64_t Mult = 1;
  static const std::pair<StringRef, uint64_t> Suffixes[] = {
      {"kib", 1024},        {"kb", 1024},        {"k", 1024},
      {"mib", 1024 * 1024}, {"mb", 1024 * 1024}, {"m", 1024 * 1024}};
  for (const auto &[Suf, M] : Suffixes) {
    if (S.take_back(Suf.size()).equals_insensitive(Suf)) {
      Mult = M;
      S = S.drop_back(Suf.size()).rtrim();
      break;
    }
  }
  uint64_t Value;
  if (S.getAsInteger(10, Value))
    return false;
  Out = Value * Mult;
  return true;
}

// Parse "N" or "MIN:MAX" (also accepts "MIN-MAX") into a flat workgroup range.
static bool parseWGRange(StringRef S, unsigned &Min, unsigned &Max) {
  S = S.trim();
  StringRef LHS, RHS;
  if (S.contains(':'))
    std::tie(LHS, RHS) = S.split(':');
  else if (S.contains('-'))
    std::tie(LHS, RHS) = S.split('-');
  else
    LHS = RHS = S;

  unsigned Lo, Hi;
  if (LHS.trim().getAsInteger(10, Lo) || RHS.trim().getAsInteger(10, Hi))
    return false;
  if (Lo == 0 || Hi == 0 || Lo > Hi)
    return false;
  Min = Lo;
  Max = Hi;
  return true;
}

static std::string formatBytes(uint64_t Bytes) {
  if (Bytes && Bytes % 1024 == 0)
    return (Twine(Bytes) + " bytes (" + Twine(Bytes / 1024) + " KiB)").str();
  return (Twine(Bytes) + " bytes").str();
}

int main(int argc, char **argv) {
  InitLLVM X(argc, argv);
  const char *ToolName = argv[0];

  cl::HideUnrelatedOptions(OccCategory);
  cl::ParseCommandLineOptions(
      argc, argv,
      "AMDGPU occupancy calculator\n\n"
      "  Prints the occupancy (waves per EU) implied by a given workgroup "
      "size,\n"
      "  VGPR/SGPR usage and LDS allocation. Unspecified fields are reported "
      "as\n"
      "  a range.\n");

  LLVMInitializeAMDGPUTargetInfo();
  LLVMInitializeAMDGPUTarget();
  LLVMInitializeAMDGPUTargetMC();

  if (MCPU.empty()) {
    WithColor::error(errs(), ToolName)
        << "no GPU specified; pass -mcpu=<gfxNNN> (e.g. -mcpu=gfx90a)\n";
    return 1;
  }

  Triple TT(Triple::normalize(TripleName));
  if (!TT.isAMDGCN()) {
    WithColor::error(errs(), ToolName)
        << "this tool only supports the AMDGPU target; got triple '" << TT.str()
        << "'\n";
    return 1;
  }

  std::string Error;
  const Target *T = TargetRegistry::lookupTarget(TT, Error);
  if (!T) {
    WithColor::error(errs(), ToolName) << Error << "\n";
    return 1;
  }

  TargetOptions Options;
  std::unique_ptr<TargetMachine> TM(T->createTargetMachine(
      TT, MCPU, MAttr, Options, std::nullopt, std::nullopt));
  if (!TM) {
    WithColor::error(errs(), ToolName)
        << "failed to create target machine for '" << MCPU << "'\n";
    return 1;
  }

  GCNSubtarget ST(TM->getTargetTriple(), std::string(TM->getTargetCPU()),
                  std::string(TM->getTargetFeatureString()),
                  *static_cast<GCNTargetMachine *>(TM.get()));

  const MCSubtargetInfo &STI = ST;

  // Parse inputs.
  unsigned WGMin = 1, WGMax = AMDGPU::IsaInfo::getMaxFlatWorkGroupSize();
  bool WGSpecified = !WGSizeStr.empty();
  if (WGSpecified && !parseWGRange(WGSizeStr, WGMin, WGMax)) {
    WithColor::error(errs(), ToolName)
        << "invalid --wg-size '" << WGSizeStr << "'\n";
    return 1;
  }

  uint64_t LDSBytes = 0;
  bool LDSSpecified = !LDSStr.empty();
  if (LDSSpecified && !parseSize(LDSStr, LDSBytes)) {
    WithColor::error(errs(), ToolName) << "invalid --lds '" << LDSStr << "'\n";
    return 1;
  }

  bool VGPRSpecified = NumVGPRs >= 0;
  bool SGPRSpecified = NumSGPRs >= 0;

  // Hardware characteristics.
  unsigned WaveSize = AMDGPU::IsaInfo::getWavefrontSize(STI);
  unsigned MaxWaves = AMDGPU::IsaInfo::getMaxWavesPerEU(STI);
  unsigned EUsPerCU = AMDGPU::IsaInfo::getEUsPerCU(STI);
  unsigned LocalMemSize = AMDGPU::IsaInfo::getLocalMemorySize(STI);
  unsigned AddrLocalMem = AMDGPU::IsaInfo::getAddressableLocalMemorySize(STI);
  unsigned AddrVGPRs =
      AMDGPU::IsaInfo::getAddressableNumVGPRs(STI, DynVGPRBlockSize);
  unsigned AddrSGPRs = ST.getAddressableNumSGPRs();
  unsigned MaxWGSize = AMDGPU::IsaInfo::getMaxFlatWorkGroupSize();

  // Warn about inputs that exceed the hardware's physical capacity: such a
  // kernel could not actually launch, so the reported occupancy is only the
  // math extrapolated past the limit.
  auto Warn = [ToolName](const Twine &Msg) {
    WithColor::warning(errs(), ToolName) << Msg << "\n";
  };
  if (LDSSpecified && LDSBytes > AddrLocalMem)
    Warn("LDS request (" + Twine(LDSBytes) +
         " bytes) exceeds addressable LDS "
         "per workgroup (" +
         Twine(AddrLocalMem) + " bytes)");
  if (VGPRSpecified && static_cast<unsigned>(NumVGPRs) > AddrVGPRs)
    Warn("VGPR request (" + Twine(static_cast<int>(NumVGPRs)) +
         ") exceeds addressable "
         "VGPRs (" +
         Twine(AddrVGPRs) + ")");
  if (SGPRSpecified && static_cast<unsigned>(NumSGPRs) > AddrSGPRs)
    Warn("SGPR request (" + Twine(static_cast<int>(NumSGPRs)) +
         ") exceeds addressable "
         "SGPRs (" +
         Twine(AddrSGPRs) + ")");
  if (WGMax > MaxWGSize)
    Warn("workgroup size (" + Twine(WGMax) +
         ") exceeds the maximum flat "
         "workgroup size (" +
         Twine(MaxWGSize) + ")");

  outs() << "llvm-calc-occupancy - AMDGPU occupancy calculator\n\n";
  outs() << "Target\n";
  outs() << format("  %-20s %s\n", "Triple:", TT.str().c_str());
  outs() << format("  %-20s %s\n",
                   "GPU (-mcpu):", std::string(TM->getTargetCPU()).c_str());
  outs() << format("  %-20s %u\n", "Wavefront size:", WaveSize);
  outs() << format("  %-20s %u (waves per SIMD, hardware limit)\n",
                   "Max waves/EU:", MaxWaves);
  outs() << format("  %-20s %u\n", "EUs (SIMDs) per CU:", EUsPerCU);
  outs() << format("  %-20s %s\n",
                   "LDS per CU:", formatBytes(LocalMemSize).c_str());
  outs() << format("  %-20s %s\n", "Addressable LDS:",
                   (formatBytes(AddrLocalMem) + " per workgroup").c_str());

  outs() << "\nInputs\n";
  if (WGSpecified) {
    if (WGMin == WGMax)
      outs() << format("  %-20s %u\n", "Workgroup size:", WGMin);
    else
      outs() << format("  %-20s %u .. %u\n", "Workgroup size:", WGMin, WGMax);
  } else {
    outs() << format("  %-20s %u .. %u (unspecified -> full range)\n",
                     "Workgroup size:", WGMin, WGMax);
  }
  if (VGPRSpecified)
    outs() << format("  %-20s %d\n",
                     "VGPRs per lane:", static_cast<int>(NumVGPRs));
  else
    outs() << format("  %-20s %s\n", "VGPRs per lane:", "unspecified");
  if (SGPRSpecified)
    outs() << format("  %-20s %d\n",
                     "SGPRs per wave:", static_cast<int>(NumSGPRs));
  else
    outs() << format("  %-20s %s\n", "SGPRs per wave:", "unspecified");
  if (LDSSpecified)
    outs() << format("  %-20s %s\n",
                     "LDS per workgroup:", formatBytes(LDSBytes).c_str());
  else
    outs() << format("  %-20s %s\n", "LDS per workgroup:", "unspecified (0)");

  // Per-constraint occupancy (all in waves/EU).
  auto [WGMinOcc, WGMaxOcc] = ST.getOccupancyWithWorkGroupSizes(
      static_cast<uint32_t>(LDSBytes), {WGMin, WGMax});
  unsigned VGPROcc =
      VGPRSpecified ? ST.getOccupancyWithNumVGPRs(NumVGPRs, DynVGPRBlockSize)
                    : MaxWaves;
  unsigned SGPROcc =
      SGPRSpecified ? ST.getOccupancyWithNumSGPRs(NumSGPRs) : MaxWaves;

  outs() << "\nPer-constraint occupancy (waves/EU)\n";
  if (WGMinOcc == WGMaxOcc)
    outs() << format("  %-20s %u\n", "Workgroup + LDS:", WGMaxOcc);
  else
    outs() << format("  %-20s %u .. %u\n", "Workgroup + LDS:", WGMinOcc,
                     WGMaxOcc);
  if (VGPRSpecified)
    outs() << format("  %-20s %u\n", "VGPRs:", VGPROcc);
  if (SGPRSpecified)
    outs() << format("  %-20s %u\n", "SGPRs:", SGPROcc);

  // Combine like GCNSubtarget::computeOccupancy.
  unsigned MaxOcc = std::min({WGMaxOcc, VGPROcc, SGPROcc});
  unsigned MinOcc = std::min(WGMinOcc, MaxOcc);

  // Identify what pins the maximum occupancy.
  SmallVector<StringRef, 3> LimitedBy;
  if (WGMaxOcc == MaxOcc)
    LimitedBy.push_back("workgroup size / LDS");
  if (VGPRSpecified && VGPROcc == MaxOcc)
    LimitedBy.push_back("VGPRs");
  if (SGPRSpecified && SGPROcc == MaxOcc)
    LimitedBy.push_back("SGPRs");

  outs() << "\nResult\n";
  if (MinOcc == MaxOcc)
    outs() << format("  %-20s %u waves/EU (%u waves/CU)\n",
                     "Occupancy:", MaxOcc, MaxOcc * EUsPerCU);
  else
    outs() << format("  %-20s %u .. %u waves/EU (%u .. %u waves/CU)\n",
                     "Occupancy:", MinOcc, MaxOcc, MinOcc * EUsPerCU,
                     MaxOcc * EUsPerCU);
  outs() << format("  %-20s %s\n",
                   "Limited by:", join(LimitedBy, ", ").c_str());

  // Hint: what would it take to gain one more wave/EU. Every factor that
  // currently pins the occupancy has to be relaxed, so list each one.
  if (MaxOcc >= MaxWaves) {
    outs() << format("  %-20s already at the hardware maximum\n", "Next step:");
  } else {
    unsigned TargetOcc = MaxOcc + 1;
    outs() << format("  %-20s reach %u waves/EU%s\n", "Next step:", TargetOcc,
                     LimitedBy.size() > 1 ? " (requires all of):" : ":");

    if (VGPRSpecified && VGPROcc == MaxOcc) {
      unsigned MaxV =
          AMDGPU::IsaInfo::getMaxNumVGPRs(STI, TargetOcc, DynVGPRBlockSize);
      outs() << format("      VGPRs <= %u (currently %d)\n", MaxV,
                       static_cast<int>(NumVGPRs));
    }
    if (SGPRSpecified && SGPROcc == MaxOcc) {
      unsigned MaxS =
          AMDGPU::IsaInfo::getMaxNumSGPRs(STI, TargetOcc, /*Addressable=*/true);
      outs() << format("      SGPRs <= %u (currently %d)\n", MaxS,
                       static_cast<int>(NumSGPRs));
    }
    if (WGMaxOcc == MaxOcc) {
      auto WGLDSOcc = [&](uint32_t LDS, unsigned Lo, unsigned Hi) {
        return ST.getOccupancyWithWorkGroupSizes(LDS, {Lo, Hi}).second;
      };
      uint32_t LDS32 = static_cast<uint32_t>(LDSBytes);
      bool Suggested = false;
      // LDS lever: largest LDS that still reaches TargetOcc at current WG.
      if (LDSSpecified && LDSBytes > 0 &&
          WGLDSOcc(0, WGMin, WGMax) >= TargetOcc) {
        uint64_t Lo = 0, Hi = LDSBytes;
        while (Lo < Hi) {
          uint64_t Mid = Lo + (Hi - Lo + 1) / 2;
          if (WGLDSOcc(static_cast<uint32_t>(Mid), WGMin, WGMax) >= TargetOcc)
            Lo = Mid;
          else
            Hi = Mid - 1;
        }
        outs() << format("      LDS <= %llu bytes (currently %llu)\n",
                         static_cast<unsigned long long>(Lo),
                         static_cast<unsigned long long>(LDSBytes));
        Suggested = true;
      }
      // Workgroup lever: largest flat workgroup size that reaches TargetOcc.
      if (WGMax > 1 && WGLDSOcc(LDS32, 1, 1) >= TargetOcc) {
        unsigned Lo = 1, Hi = WGMax;
        while (Lo < Hi) {
          unsigned Mid = Lo + (Hi - Lo + 1) / 2;
          if (WGLDSOcc(LDS32, Mid, Mid) >= TargetOcc)
            Lo = Mid;
          else
            Hi = Mid - 1;
        }
        if (Lo < WGMax) {
          outs() << format("      workgroup size <= %u (currently %u)\n", Lo,
                           WGMax);
          Suggested = true;
        }
      }
      if (!Suggested)
        outs() << "      reduce workgroup size and/or LDS\n";
    }
  }

  if (ShowLimits) {
    outs() << "\nPer-occupancy register limits (max regs to still reach "
              "each level)\n";
    outs() << format("  %-14s %-14s %-14s\n", "Occupancy", "Max VGPRs",
                     "Max SGPRs");
    for (unsigned Occ = MaxWaves; Occ >= 1; --Occ) {
      unsigned MaxV =
          AMDGPU::IsaInfo::getMaxNumVGPRs(STI, Occ, DynVGPRBlockSize);
      unsigned MaxS =
          AMDGPU::IsaInfo::getMaxNumSGPRs(STI, Occ, /*Addressable=*/true);
      outs() << format("  %-14u %-14u %-14u\n", Occ, MaxV, MaxS);
    }
  }

  return 0;
}
