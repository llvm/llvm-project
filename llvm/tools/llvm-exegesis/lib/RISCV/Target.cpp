//===-- Target.cpp ----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../Target.h"
#include "../ParallelSnippetGenerator.h"
#include "../SerialSnippetGenerator.h"
#include "../SnippetGenerator.h"
#include "MCTargetDesc/RISCVBaseInfo.h"
#include "MCTargetDesc/RISCVMatInt.h"
#include "RISCV.h"
#include "RISCVExegesisPasses.h"
#include "RISCVInstrInfo.h"
#include "RISCVRegisterInfo.h"
#include "RISCVSubtarget.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/raw_ostream.h"

#include <array>

#include <linux/perf_event.h>

#define GET_AVAILABLE_OPCODE_CHECKER
#include "RISCVGenInstrInfo.inc"

namespace RVVPseudoTables {
using namespace llvm;
using namespace llvm::RISCV;

struct PseudoInfo {
  uint16_t Pseudo;
  uint16_t BaseInstr;
  uint8_t VLMul;
  uint8_t SEW;
};

struct RISCVMaskedPseudoInfo {
  uint16_t MaskedPseudo;
  uint16_t UnmaskedPseudo;
  uint8_t MaskOpIdx;
};

#define GET_RISCVVInversePseudosTable_IMPL
#define GET_RISCVVInversePseudosTable_DECL
#define GET_RISCVMaskedPseudosTable_DECL
#define GET_RISCVMaskedPseudosTable_IMPL
#include "RISCVGenSearchableTables.inc"

} // namespace RVVPseudoTables

namespace llvm {
namespace exegesis {

static cl::opt<bool>
    OnlyUsesVLMAXForVL("riscv-vlmax-for-vl",
                       cl::desc("Only enumerate VLMAX for VL operand"),
                       cl::init(false), cl::Hidden);

static cl::opt<bool>
    EnumerateRoundingModes("riscv-enumerate-rounding-modes",
                           cl::desc("Enumerate different FRM and VXRM"),
                           cl::init(true), cl::Hidden);

static cl::opt<std::string>
    FilterConfig("riscv-filter-config",
                 cl::desc("Show only the configs matching this regex"),
                 cl::init(""), cl::Hidden);

#include "RISCVGenExegesis.inc"

namespace {

static perf_event_attr *createPerfEventAttr(unsigned Type, uint64_t Config) {
  auto *PEA = new perf_event_attr();
  memset(PEA, 0, sizeof(perf_event_attr));
  PEA->type = Type;
  PEA->size = sizeof(perf_event_attr);
  PEA->config = Config;
  PEA->disabled = 1;
  PEA->exclude_kernel = 1;
  PEA->exclude_hv = 1;
  return PEA;
}

struct RISCVPerfEvent : public pfm::PerfEvent {
  explicit RISCVPerfEvent(StringRef PfmEventString)
      : pfm::PerfEvent(PfmEventString) {
    FullQualifiedEventString = EventString;

    if (EventString == "CYCLES" || EventString == "CPU_CYCLES")
      Attr = createPerfEventAttr(PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES);
  }
};

template <class BaseT> class RVVSnippetGenerator : public BaseT {
  static void printRoundingMode(raw_ostream &OS, unsigned Val, bool UsesVXRM) {
    static const char *const FRMNames[] = {"rne", "rtz", "rdn", "rup",
                                           "rmm", "N/A", "N/A", "dyn"};
    static const char *const VXRMNames[] = {"rnu", "rne", "rdn", "rod"};

    if (UsesVXRM) {
      assert(Val < 4);
      OS << VXRMNames[Val];
    } else {
      assert(Val != 5 && Val != 6);
      OS << FRMNames[Val];
    }
  }

  static constexpr unsigned MinSEW = 8;
  // ELEN is basically SEW_max.
  static constexpr unsigned ELEN = 64;

  // We can't know the real min/max VLEN w/o a Function, so we're
  // using the VLen from Zvl.
  unsigned ZvlVLen = 32;

  /// Mask for registers that are NOT standalone registers like X0 and V0
  BitVector AggregateRegisters;

  // Returns true when opcode is available in any of the FBs.
  static bool
  isOpcodeAvailableIn(unsigned Opcode,
                      ArrayRef<RISCV_MC::SubtargetFeatureBits> FBs) {
    FeatureBitset RequiredFeatures = RISCV_MC::computeRequiredFeatures(Opcode);
    for (uint8_t FB : FBs) {
      if (RequiredFeatures[FB])
        return true;
    }
    return false;
  }

  static bool isRVVFloatingPointOp(unsigned Opcode) {
    return isOpcodeAvailableIn(Opcode,
                               {RISCV_MC::Feature_HasVInstructionsAnyFBit});
  }

  // Get the element group width of each vector cryptor extension.
  static unsigned getZvkEGWSize(unsigned Opcode, unsigned SEW) {
    using namespace RISCV_MC;
    if (isOpcodeAvailableIn(Opcode, {Feature_HasStdExtZvkgBit,
                                     Feature_HasStdExtZvknedBit,
                                     Feature_HasStdExtZvksedBit}))
      return 128U;
    else if (isOpcodeAvailableIn(Opcode, {Feature_HasStdExtZvkshBit}))
      return 256U;
    else if (isOpcodeAvailableIn(Opcode, {Feature_HasStdExtZvknhaOrZvknhbBit}))
      // In Zvknh[ab], when SEW=64 is used (i.e. Zvknhb), EGW is 256.
      // Otherwise it's 128.
      return SEW == 64 ? 256U : 128U;

    llvm_unreachable("Unsupported opcode");
  }

  // A handy utility to multiply or divide an integer by LMUL.
  template <typename T> static T multiplyLMul(T Val, RISCVII::VLMUL LMul) {
    // Fractional
    if (LMul >= RISCVII::LMUL_F8)
      return Val >> (8 - LMul);
    else
      return Val << LMul;
  }

  /// Return the denominator of the fractional (i.e. the `x` in .vfx suffix) or
  /// nullopt if BaseOpcode is not a vector sext/zext.
  static std::optional<unsigned> isRVVSignZeroExtend(unsigned BaseOpcode) {
    switch (BaseOpcode) {
    case RISCV::VSEXT_VF2:
    case RISCV::VZEXT_VF2:
      return 2;
    case RISCV::VSEXT_VF4:
    case RISCV::VZEXT_VF4:
      return 4;
    case RISCV::VSEXT_VF8:
    case RISCV::VZEXT_VF8:
      return 8;
    default:
      return std::nullopt;
    }
  }

  void annotateWithVType(const CodeTemplate &CT, const Instruction &Instr,
                         unsigned BaseOpcode,
                         const BitVector &ForbiddenRegisters,
                         std::vector<CodeTemplate> &Result) const;

public:
  RVVSnippetGenerator(const LLVMState &State,
                      const SnippetGenerator::Options &Opts)
      : BaseT(State, Opts),
        AggregateRegisters(State.getRegInfo().getNumRegs(), /*initVal=*/true) {
    // Initialize standalone registers mask.
    const MCRegisterInfo &RegInfo = State.getRegInfo();
    const unsigned StandaloneRegClasses[] = {
        RISCV::GPRRegClassID, RISCV::FPR16RegClassID, RISCV::VRRegClassID};

    for (unsigned RegClassID : StandaloneRegClasses)
      for (unsigned Reg : RegInfo.getRegClass(RegClassID)) {
        AggregateRegisters.reset(Reg);
      }

    // Initialize the ZvlVLen.
    const MCSubtargetInfo &STI = State.getSubtargetInfo();
    std::string ZvlQuery;
    for (unsigned I = 5U, Size = (1 << I); I < 17U; ++I, Size <<= 1) {
      ZvlQuery = "+zvl";
      raw_string_ostream SS(ZvlQuery);
      SS << Size << "b";
      if (STI.checkFeatures(SS.str()) && ZvlVLen < Size)
        ZvlVLen = Size;
    }
  }

  Expected<std::vector<CodeTemplate>>
  generateCodeTemplates(InstructionTemplate Variant,
                        const BitVector &ForbiddenRegisters) const override;
};

static bool isMaskedSibiling(unsigned MaskedOp, unsigned UnmaskedOp) {
  const auto *RVVMasked = RVVPseudoTables::getMaskedPseudoInfo(MaskedOp);
  return RVVMasked && RVVMasked->UnmaskedPseudo == UnmaskedOp;
}

// There are primarily two kinds of opcodes that are not eligible
// in a serial snippet:
// (1) Only has a single use operand that can not be overlap with
// the def operand.
// (2) The register file of the only use operand is different from
// that of the def operand. For instance, use operand is vector and
// the result is a scalar.
static bool isIneligibleOfSerialSnippets(unsigned BaseOpcode,
                                         const Instruction &I) {
  if (llvm::any_of(I.Operands,
                   [](const Operand &Op) { return Op.isEarlyClobber(); }))
    return true;

  switch (BaseOpcode) {
  case RISCV::VCOMPRESS_VM:
  case RISCV::VCPOP_M:
  case RISCV::VCPOP_V:
  case RISCV::VRGATHEREI16_VV:
  case RISCV::VRGATHER_VI:
  case RISCV::VRGATHER_VV:
  case RISCV::VRGATHER_VX:
  case RISCV::VSLIDE1UP_VX:
  case RISCV::VSLIDEUP_VI:
  case RISCV::VSLIDEUP_VX:
  // The truncate instructions that arraive here are those who cannot
  // have any overlap between source and dest at all (i.e.
  // those whoe don't satisfy condition 2 and 3 in RVV spec
  // 5.2).
  case RISCV::VNCLIPU_WI:
  case RISCV::VNCLIPU_WV:
  case RISCV::VNCLIPU_WX:
  case RISCV::VNCLIP_WI:
  case RISCV::VNCLIP_WV:
  case RISCV::VNCLIP_WX:
    return true;
  default:
    return false;
  }
}

static bool isZvfhminZvfbfminOpcodes(unsigned BaseOpcode) {
  switch (BaseOpcode) {
  case RISCV::VFNCVT_F_F_W:
  case RISCV::VFWCVT_F_F_V:
  case RISCV::VFNCVTBF16_F_F_W:
  case RISCV::VFWCVTBF16_F_F_V:
    return true;
  default:
    return false;
  }
}

static bool isVectorReduction(unsigned BaseOpcode) {
  switch (BaseOpcode) {
  case RISCV::VREDAND_VS:
  case RISCV::VREDMAXU_VS:
  case RISCV::VREDMAX_VS:
  case RISCV::VREDMINU_VS:
  case RISCV::VREDMIN_VS:
  case RISCV::VREDOR_VS:
  case RISCV::VREDSUM_VS:
  case RISCV::VREDXOR_VS:
  case RISCV::VWREDSUMU_VS:
  case RISCV::VWREDSUM_VS:
  case RISCV::VFREDMAX_VS:
  case RISCV::VFREDMIN_VS:
  case RISCV::VFREDOSUM_VS:
  case RISCV::VFREDUSUM_VS:
    return true;
  default:
    return false;
  }
}

template <class BaseT>
void RVVSnippetGenerator<BaseT>::annotateWithVType(
    const CodeTemplate &OrigCT, const Instruction &Instr, unsigned BaseOpcode,
    const BitVector &ForbiddenRegisters,
    std::vector<CodeTemplate> &Result) const {
  const MCSubtargetInfo &STI = SnippetGenerator::State.getSubtargetInfo();
  unsigned VPseudoOpcode = Instr.getOpcode();

  bool IsSerial = std::is_same_v<BaseT, SerialSnippetGenerator>;

  const MCInstrDesc &MIDesc = Instr.Description;
  const uint64_t TSFlags = MIDesc.TSFlags;

  RISCVII::VLMUL VLMul = RISCVII::getLMul(TSFlags);

  const size_t StartingResultSize = Result.size();

  SmallPtrSet<const Operand *, 4> VTypeOperands;
  std::optional<AliasingConfigurations> SelfAliasing;
  // Exegesis see instructions with tied operands being inherently serial.
  // But for RVV instructions, those tied operands are passthru rather
  // than real read operands. So we manually put dependency between
  // destination (i.e. def) and any of the non-tied/SEW/policy/AVL/RM
  // operands.
  auto assignSerialRVVOperands = [&, this](InstructionTemplate &IT) {
    // Initialize SelfAliasing on first use.
    if (!SelfAliasing.has_value()) {
      BitVector ExcludeRegs = ForbiddenRegisters;
      ExcludeRegs |= AggregateRegisters;
      SelfAliasing = AliasingConfigurations(Instr, Instr, ExcludeRegs);
      bool EmptyUses = false;
      for (auto &ARO : SelfAliasing->Configurations) {
        auto &Uses = ARO.Uses;
        for (auto ROA = Uses.begin(); ROA != Uses.end();) {
          const Operand *Op = ROA->Op;
          // Exclude tied operand(s).
          if (Op->isTied()) {
            ROA = Uses.erase(ROA);
            continue;
          }

          // Special handling for reduction operations: for a given reduction
          // `vredop vd, vs2, vs1`, we don't want vd to be aliased with vs1
          // since we're only reading `vs1[0]` and many implementations
          // optimize for this case (e.g. chaining). Instead, we're forcing
          // it to create alias between vd and vs2.
          if (isVectorReduction(BaseOpcode) &&
              // vs1's operand index is always 3.
              Op->getIndex() == 3) {
            ROA = Uses.erase(ROA);
            continue;
          }

          // Exclude any special operands like SEW and VL -- we've already
          // assigned values to them.
          if (VTypeOperands.count(Op)) {
            ROA = Uses.erase(ROA);
            continue;
          }
          ++ROA;
        }

        // If any of the use operand candidate lists is empty, there is
        // no point to assign self aliasing registers.
        if (Uses.empty()) {
          EmptyUses = true;
          break;
        }
      }
      if (EmptyUses)
        SelfAliasing->Configurations.clear();
    }

    // This is a self aliasing instruction so defs and uses are from the same
    // instance, hence twice IT in the following call.
    if (!SelfAliasing->empty() && !SelfAliasing->hasImplicitAliasing())
      setRandomAliasing(*SelfAliasing, IT, IT);
  };

  // We are going to create a CodeTemplate (configuration) for each supported
  // SEW, policy, and VL.
  // FIXME: Account for EEW and EMUL.
  SmallVector<std::optional<unsigned>, 4> Log2SEWs;
  SmallVector<std::optional<unsigned>, 4> Policies;
  SmallVector<std::optional<int>, 3> AVLs;
  SmallVector<std::optional<unsigned>, 8> RoundingModes;

  bool HasSEWOp = RISCVII::hasSEWOp(TSFlags);
  bool HasPolicyOp = RISCVII::hasVecPolicyOp(TSFlags);
  bool HasVLOp = RISCVII::hasVLOp(TSFlags);
  bool HasRMOp = RISCVII::hasRoundModeOp(TSFlags);
  bool UsesVXRM = RISCVII::usesVXRM(TSFlags);

  if (HasSEWOp) {
    VTypeOperands.insert(&Instr.Operands[RISCVII::getSEWOpNum(MIDesc)]);

    SmallVector<unsigned, 4> SEWCandidates;

    // (RVV spec 3.4.2) For fractional LMUL, the supported SEW are between
    // [SEW_min, LMUL * ELEN].
    unsigned SEWUpperBound =
        VLMul >= RISCVII::LMUL_F8 ? multiplyLMul(ELEN, VLMul) : ELEN;
    for (unsigned SEW = MinSEW; SEW <= SEWUpperBound; SEW <<= 1) {
      SEWCandidates.push_back(SEW);

      // Some scheduling classes already integrate SEW; only put
      // their corresponding SEW values at the SEW operands.
      // NOTE: It is imperative to put this condition in the front, otherwise
      // it is tricky and difficult to know if there is an integrated
      // SEW after other rules are applied to filter the candidates.
      const auto *RVVBase =
          RVVPseudoTables::getBaseInfo(BaseOpcode, VLMul, SEW);
      if (RVVBase && (RVVBase->Pseudo == VPseudoOpcode ||
                      isMaskedSibiling(VPseudoOpcode, RVVBase->Pseudo) ||
                      isMaskedSibiling(RVVBase->Pseudo, VPseudoOpcode))) {
        // There is an integrated SEW, remove all but the SEW pushed last.
        SEWCandidates.erase(SEWCandidates.begin(), SEWCandidates.end() - 1);
        break;
      }
    }

    // Filter out some candidates.
    for (auto SEW = SEWCandidates.begin(); SEW != SEWCandidates.end();) {
      // For floating point operations, only select SEW of the supported FLEN.
      if (isRVVFloatingPointOp(VPseudoOpcode)) {
        bool Supported = false;
        Supported |= isZvfhminZvfbfminOpcodes(BaseOpcode) && *SEW == 16;
        Supported |= STI.hasFeature(RISCV::FeatureStdExtZvfh) && *SEW == 16;
        Supported |= STI.hasFeature(RISCV::FeatureStdExtF) && *SEW == 32;
        Supported |= STI.hasFeature(RISCV::FeatureStdExtD) && *SEW == 64;
        if (!Supported) {
          SEW = SEWCandidates.erase(SEW);
          continue;
        }
      }

      // The EEW for source operand in VSEXT and VZEXT is a fractional
      // of the SEW, hence only SEWs that will lead to valid EEW are allowed.
      if (auto Frac = isRVVSignZeroExtend(BaseOpcode))
        if (*SEW / *Frac < MinSEW) {
          SEW = SEWCandidates.erase(SEW);
          continue;
        }

      // Most vector crypto 1.0 instructions only work on SEW=32.
      using namespace RISCV_MC;
      if (isOpcodeAvailableIn(BaseOpcode, {Feature_HasStdExtZvkgBit,
                                           Feature_HasStdExtZvknedBit,
                                           Feature_HasStdExtZvknhaOrZvknhbBit,
                                           Feature_HasStdExtZvksedBit,
                                           Feature_HasStdExtZvkshBit})) {
        if (*SEW != 32)
          // Zvknhb support SEW=64 as well.
          if (*SEW != 64 || !STI.hasFeature(RISCV::FeatureStdExtZvknhb) ||
              !isOpcodeAvailableIn(BaseOpcode,
                                   {Feature_HasStdExtZvknhaOrZvknhbBit})) {
            SEW = SEWCandidates.erase(SEW);
            continue;
          }

        // We're also enforcing the requirement of `LMUL * VLEN >= EGW` here,
        // because some of the extensions have SEW-dependant EGW.
        unsigned EGW = getZvkEGWSize(BaseOpcode, *SEW);
        if (multiplyLMul(ZvlVLen, VLMul) < EGW) {
          SEW = SEWCandidates.erase(SEW);
          continue;
        }
      }

      ++SEW;
    }

    // We're not going to produce any result with zero SEW candidate.
    if (SEWCandidates.empty())
      return;

    for (unsigned SEW : SEWCandidates)
      Log2SEWs.push_back(SEW == 8 ? 0 : Log2_32(SEW));
  } else {
    Log2SEWs.push_back(std::nullopt);
  }

  if (HasPolicyOp) {
    VTypeOperands.insert(&Instr.Operands[RISCVII::getVecPolicyOpNum(MIDesc)]);

    Policies = {0, RISCVII::TAIL_AGNOSTIC, RISCVII::MASK_AGNOSTIC,
                (RISCVII::TAIL_AGNOSTIC | RISCVII::MASK_AGNOSTIC)};
  } else {
    Policies.push_back(std::nullopt);
  }

  if (HasVLOp) {
    VTypeOperands.insert(&Instr.Operands[RISCVII::getVLOpNum(MIDesc)]);

    if (OnlyUsesVLMAXForVL)
      AVLs.push_back(-1);
    else
      AVLs = {// 5-bit immediate value
              1,
              // VLMAX
              -1,
              // Non-X0 register
              0};
  } else {
    AVLs.push_back(std::nullopt);
  }

  if (HasRMOp) {
    VTypeOperands.insert(&Instr.Operands[RISCVII::getVLOpNum(MIDesc) - 1]);

    // If we're not enumerating all rounding modes,
    // use zero (rne in FRM and rnu in VXRM) as the default
    // mode.
    RoundingModes = {0U};
    if (EnumerateRoundingModes) {
      RoundingModes.append({1, 2, 3});
      if (!UsesVXRM)
        // FRM values 5 and 6 are currently reserved.
        RoundingModes.append({4, 7});
    }
  } else {
    RoundingModes = {std::nullopt};
  }

  std::set<std::tuple<std::optional<unsigned>, std::optional<int>,
                      std::optional<unsigned>, std::optional<unsigned>>>
      Combinations;
  for (auto AVL : AVLs) {
    for (auto Log2SEW : Log2SEWs)
      for (auto Policy : Policies) {
        for (auto RM : RoundingModes)
          Combinations.insert(std::make_tuple(RM, AVL, Log2SEW, Policy));
      }
  }

  std::string ConfigStr;
  SmallVector<std::pair<const Operand *, MCOperand>, 4> ValueAssignments;
  for (const auto &[RM, AVL, Log2SEW, Policy] : Combinations) {
    InstructionTemplate IT(&Instr);

    ListSeparator LS;
    ConfigStr = "vtype = {";
    raw_string_ostream SS(ConfigStr);

    ValueAssignments.clear();

    if (RM) {
      const Operand &Op = Instr.Operands[RISCVII::getVLOpNum(MIDesc) - 1];
      ValueAssignments.push_back({&Op, MCOperand::createImm(*RM)});
      printRoundingMode(SS << LS << (UsesVXRM ? "VXRM" : "FRM") << ": ", *RM,
                        UsesVXRM);
    }

    if (AVL) {
      MCOperand OpVal;
      if (*AVL < 0) {
        // VLMAX
        OpVal = MCOperand::createImm(-1);
        SS << LS << "AVL: VLMAX";
      } else if (*AVL == 0) {
        // A register holding AVL.
        // TODO: Generate a random register.
        OpVal = MCOperand::createReg(RISCV::X5);
        OpVal.print(SS << LS << "AVL: ");
      } else {
        // A 5-bit immediate.
        // The actual value assignment is deferred to
        // RISCVExegesisTarget::randomizeTargetMCOperand.
        SS << LS << "AVL: simm5";
      }
      if (OpVal.isValid()) {
        const Operand &Op = Instr.Operands[RISCVII::getVLOpNum(MIDesc)];
        ValueAssignments.push_back({&Op, OpVal});
      }
    }

    if (Log2SEW) {
      const Operand &Op = Instr.Operands[RISCVII::getSEWOpNum(MIDesc)];
      ValueAssignments.push_back({&Op, MCOperand::createImm(*Log2SEW)});
      SS << LS << "SEW: e" << (*Log2SEW ? 1 << *Log2SEW : 8);
    }

    if (Policy) {
      const Operand &Op = Instr.Operands[RISCVII::getVecPolicyOpNum(MIDesc)];
      ValueAssignments.push_back({&Op, MCOperand::createImm(*Policy)});
      SS << LS << "Policy: " << (*Policy & RISCVII::TAIL_AGNOSTIC ? "ta" : "tu")
         << "/" << (*Policy & RISCVII::MASK_AGNOSTIC ? "ma" : "mu");
    }

    SS << "}";

    // Filter out some configurations, if needed.
    if (!FilterConfig.empty()) {
      if (!Regex(FilterConfig).match(ConfigStr))
        continue;
    }

    CodeTemplate CT = OrigCT.clone();
    CT.Config = std::move(ConfigStr);
    for (InstructionTemplate &IT : CT.Instructions) {
      if (IsSerial) {
        // Reset this template's value assignments and do it
        // ourselves.
        IT = InstructionTemplate(&Instr);
        assignSerialRVVOperands(IT);
      }

      for (const auto &[Op, OpVal] : ValueAssignments)
        IT.getValueFor(*Op) = OpVal;
    }
    Result.push_back(std::move(CT));
    if (Result.size() - StartingResultSize >=
        SnippetGenerator::Opts.MaxConfigsPerOpcode)
      return;
  }
}

template <class BaseT>
Expected<std::vector<CodeTemplate>>
RVVSnippetGenerator<BaseT>::generateCodeTemplates(
    InstructionTemplate Variant, const BitVector &ForbiddenRegisters) const {
  const Instruction &Instr = Variant.getInstr();

  bool IsSerial = std::is_same_v<BaseT, SerialSnippetGenerator>;

  unsigned BaseOpcode = RISCV::getRVVMCOpcode(Instr.getOpcode());

  // Bail out ineligible opcodes before generating base code templates since
  // the latter is quite expensive.
  if (IsSerial && BaseOpcode && isIneligibleOfSerialSnippets(BaseOpcode, Instr))
    return std::vector<CodeTemplate>{};

  auto BaseCodeTemplates =
      BaseT::generateCodeTemplates(Variant, ForbiddenRegisters);
  if (!BaseCodeTemplates)
    return BaseCodeTemplates.takeError();

  // We only specialize for RVVPseudo here
  if (!BaseOpcode)
    return BaseCodeTemplates;

  std::vector<CodeTemplate> ExpandedTemplates;
  for (const auto &BaseCT : *BaseCodeTemplates)
    annotateWithVType(BaseCT, Instr, BaseOpcode, ForbiddenRegisters,
                      ExpandedTemplates);

  return ExpandedTemplates;
}

// NOTE: Alternatively, we can use BitVector here, but the number of RVV opcodes
// is just a small portion of the entire opcode space, so I thought it would be
// a waste of space to use BitVector.
static SmallSet<unsigned, 16> RVVOpcodesWithPseudos;

class ExegesisRISCVTarget : public ExegesisTarget {
public:
  ExegesisRISCVTarget()
      : ExegesisTarget(RISCVCpuPfmCounters, RISCV_MC::isOpcodeAvailable) {}

private:
  bool isOpcodeSupported(const MCInstrDesc &Desc) const override {
    switch (Desc.getOpcode()) {
    case RISCV::PseudoVSETIVLI:
    case RISCV::PseudoVSETVLI:
    case RISCV::PseudoVSETVLIX0:
    case RISCV::VSETIVLI:
    case RISCV::VSETVLI:
    case RISCV::VSETVL:
      return false;
    default:
      break;
    }

    // We want to support all the RVV pseudos.
    if (unsigned Opcode = RISCV::getRVVMCOpcode(Desc.getOpcode())) {
      RVVOpcodesWithPseudos.insert(Opcode);
      return true;
    }

    // We don't want to support RVV instructions that depend on VTYPE, because
    // those instructions by themselves don't carry any additional information
    // for us to setup the proper VTYPE environment via VSETVL instructions.
    // FIXME: Ideally, we should have a list of such RVV instructions...except
    // we don't have, hence we use an ugly trick here to memorize the
    // corresponding MC opcodes of the RVV pseudo we have processed previously.
    // This works most of the time because RVV pseudo opcodes are placed before
    // any other RVV opcodes. Of course this doesn't work if we're asked to
    // benchmark only a certain subset of opcodes.
    if (RVVOpcodesWithPseudos.count(Desc.getOpcode()))
      return false;

    return ExegesisTarget::isOpcodeSupported(Desc);
  }

  Error
  randomizeTargetMCOperand(const Instruction &Instr, const Variable &Var,
                           MCOperand &AssignedValue,
                           const BitVector &ForbiddenRegs) const override {
    const Operand &Op = Instr.getPrimaryOperand(Var);
    switch (Op.getExplicitOperandInfo().OperandType) {
    case RISCVOp::OPERAND_SIMM5:
      // 5-bit signed immediate value.
      AssignedValue = MCOperand::createImm(randomIndex(31) - 16);
      return Error::success();
    case RISCVOp::OPERAND_AVL:
    case RISCVOp::OPERAND_UIMM5:
      // 5-bit unsigned immediate value.
      AssignedValue = MCOperand::createImm(randomIndex(31));
      return Error::success();
    default:
      break;
    }
    return make_error<Failure>(
        Twine("unimplemented operand type ")
            .concat(std::to_string(Op.getExplicitOperandInfo().OperandType)));
  }

  static std::vector<MCInst> loadIntImmediate(const MCSubtargetInfo &STI,
                                              unsigned Reg,
                                              const APInt &Value) {
    // Lower to materialization sequence.
    RISCVMatInt::InstSeq Seq =
        RISCVMatInt::generateInstSeq(Value.getSExtValue(), STI);
    assert(!Seq.empty());

    Register DstReg = Reg;
    Register SrcReg = RISCV::X0;

    std::vector<MCInst> Insts;
    for (const RISCVMatInt::Inst &Inst : Seq) {
      switch (Inst.getOpndKind()) {
      case RISCVMatInt::Imm:
        Insts.emplace_back(MCInstBuilder(Inst.getOpcode())
                               .addReg(DstReg)
                               .addImm(Inst.getImm()));
        break;
      case RISCVMatInt::RegX0:
        Insts.emplace_back(MCInstBuilder(Inst.getOpcode())
                               .addReg(DstReg)
                               .addReg(SrcReg)
                               .addReg(RISCV::X0));
        break;
      case RISCVMatInt::RegReg:
        Insts.emplace_back(MCInstBuilder(Inst.getOpcode())
                               .addReg(DstReg)
                               .addReg(SrcReg)
                               .addReg(SrcReg));
        break;
      case RISCVMatInt::RegImm:
        Insts.emplace_back(MCInstBuilder(Inst.getOpcode())
                               .addReg(DstReg)
                               .addReg(SrcReg)
                               .addImm(Inst.getImm()));
        break;
      }

      // Only the first instruction has X0 as its source.
      SrcReg = DstReg;
    }
    return Insts;
  }

  // Note that we assume the given APInt is an integer rather than a bit-casted
  // floating point value.
  static std::vector<MCInst> loadFPImmediate(unsigned FLen,
                                             const MCSubtargetInfo &STI,
                                             unsigned Reg, const APInt &Value) {
    // Try FLI from the Zfa extension.
    if (STI.hasFeature(RISCV::FeatureStdExtZfa)) {
      APFloat FloatVal(FLen == 32 ? APFloat::IEEEsingle()
                                  : APFloat::IEEEdouble());
      if (FloatVal.convertFromAPInt(Value, /*IsSigned=*/Value.isSignBitSet(),
                                    APFloat::rmNearestTiesToEven) ==
          APFloat::opOK) {
        int Idx = RISCVLoadFPImm::getLoadFPImm(FloatVal);
        if (Idx >= 0)
          return {MCInstBuilder(FLen == 32 ? RISCV::FLI_S : RISCV::FLI_D)
                      .addReg(Reg)
                      .addImm(static_cast<uint64_t>(Idx))};
      }
    }

    // Otherwise, move the value to a GPR (t0) first.
    assert(Reg != RISCV::X5);
    auto ImmSeq = loadIntImmediate(STI, RISCV::X5, Value);

    // Then, use FCVT.
    unsigned Opcode;
    if (FLen == 32)
      Opcode = Value.getBitWidth() <= 32 ? RISCV::FCVT_S_W : RISCV::FCVT_S_L;
    else
      Opcode = Value.getBitWidth() <= 32 ? RISCV::FCVT_D_W : RISCV::FCVT_D_L;
    ImmSeq.emplace_back(
        MCInstBuilder(Opcode).addReg(Reg).addReg(RISCV::X5).addImm(
            RISCVFPRndMode::RNE));

    return ImmSeq;
  }

  std::vector<MCInst> setRegTo(const MCSubtargetInfo &STI, unsigned Reg,
                               const APInt &Value) const override {
    if (Reg == RISCV::X0) {
      if (Value == 0U)
        // NOP
        return {MCInstBuilder(RISCV::ADDI)
                    .addReg(RISCV::X0)
                    .addReg(RISCV::X0)
                    .addImm(0U)};
      errs() << "Cannot write non-zero values to X0\n";
      return {};
    }

    if (RISCV::GPRNoX0RegClass.contains(Reg))
      return loadIntImmediate(STI, Reg, Value);
    if (RISCV::FPR32RegClass.contains(Reg) &&
        STI.hasFeature(RISCV::FeatureStdExtF))
      return loadFPImmediate(32, STI, Reg, Value);
    if (RISCV::FPR64RegClass.contains(Reg) &&
        STI.hasFeature(RISCV::FeatureStdExtD))
      return loadFPImmediate(64, STI, Reg, Value);
    return {};
  }

  RegisterValue assignInitialRegisterValue(const Instruction &I,
                                           const Operand &Op,
                                           unsigned Reg) const override {
    // If this is a register AVL, we don't want to assign 0 or VLMAX VL.
    if (Op.isExplicit() &&
        Op.getExplicitOperandInfo().OperandType == RISCVOp::OPERAND_AVL) {
      // Assume VLEN is 128 here.
      constexpr unsigned VLEN = 128;
      // VLMAX equals to VLEN since
      // VLMAX = VLEN / <smallest SEW = 8> * <largest LMUL = 8>.
      return RegisterValue{Reg, APInt(32, randomIndex(VLEN - 4) + 2)};
    }

    switch (I.getOpcode()) {
    // We don't want divided-by-zero for these opcodes.
    case RISCV::DIV:
    case RISCV::DIVU:
    case RISCV::DIVW:
    case RISCV::DIVUW:
    case RISCV::REM:
    case RISCV::REMU:
    case RISCV::REMW:
    case RISCV::REMUW:
    // Multiplications and its friends are not really interestings
    // when they're multiplied by zero.
    case RISCV::MUL:
    case RISCV::MULH:
    case RISCV::MULHSU:
    case RISCV::MULHU:
    case RISCV::MULW:
    case RISCV::CPOP:
    case RISCV::CPOPW:
      return RegisterValue{Reg, APInt(32, randomIndex(INT32_MAX - 1) + 1)};
    default:
      return ExegesisTarget::assignInitialRegisterValue(I, Op, Reg);
    }
  }

  bool matchesArch(Triple::ArchType Arch) const override {
    return Arch == Triple::riscv32 || Arch == Triple::riscv64;
  }

  unsigned getDefaultLoopCounterRegister(const Triple &TT) const override {
    return RISCV::X5;
  }

  void decrementLoopCounterAndJump(MachineBasicBlock &MBB,
                                   MachineBasicBlock &TargetMBB,
                                   const MCInstrInfo &MII,
                                   unsigned LoopRegister) const override {
    MIMetadata MIMD;
    BuildMI(MBB, MBB.end(), MIMD, MII.get(RISCV::ADDI), LoopRegister)
        .addUse(LoopRegister)
        .addImm(-1);
    BuildMI(MBB, MBB.end(), MIMD, MII.get(RISCV::BNE))
        .addUse(LoopRegister)
        .addUse(RISCV::X0)
        .addMBB(&TargetMBB);
  }

  std::unique_ptr<SnippetGenerator> createSerialSnippetGenerator(
      const LLVMState &State,
      const SnippetGenerator::Options &Opts) const override {
    return std::make_unique<RVVSnippetGenerator<SerialSnippetGenerator>>(State,
                                                                         Opts);
  }

  std::unique_ptr<SnippetGenerator> createParallelSnippetGenerator(
      const LLVMState &State,
      const SnippetGenerator::Options &Opts) const override {
    return std::make_unique<RVVSnippetGenerator<ParallelSnippetGenerator>>(
        State, Opts);
  }

  Expected<std::unique_ptr<pfm::CounterGroup>>
  createCounter(StringRef CounterName, const LLVMState &,
                ArrayRef<const char *> ValidationCounters,
                const pid_t ProcessID) const override {
    auto Event = static_cast<pfm::PerfEvent>(RISCVPerfEvent(CounterName));
    if (!Event.valid())
      return llvm::make_error<Failure>(
          llvm::Twine("Unable to create counter with name '")
              .concat(CounterName)
              .concat("'"));

    std::vector<pfm::PerfEvent> ValidationEvents;
    for (const char *ValCounterName : ValidationCounters) {
      ValidationEvents.emplace_back(ValCounterName);
      if (!ValidationEvents.back().valid())
        return llvm::make_error<Failure>(
            llvm::Twine("Unable to create validation counter with name '")
                .concat(ValCounterName)
                .concat("'"));
    }

    return std::make_unique<pfm::CounterGroup>(
        std::move(Event), std::move(ValidationEvents), ProcessID);
  }

  void addTargetSpecificPasses(PassManagerBase &PM) const override {
    // Turn AVL operand of physical registers into virtual registers.
    PM.add(exegesis::createRISCVPreprocessingPass());
    PM.add(createRISCVInsertVSETVLIPass());
    // Setting up the correct FRM.
    PM.add(createRISCVInsertReadWriteCSRPass());
    PM.add(createRISCVInsertWriteVXRMPass());
    // This will assign physical register to the result of VSETVLI instructions
    // that produce VLMAX.
    PM.add(exegesis::createRISCVPostprocessingPass());
    // PseudoRET will be expanded by RISCVAsmPrinter; we have to expand
    // PseudoMovImm with RISCVPostRAExpandPseudoPass though.
    PM.add(createRISCVPostRAExpandPseudoPass());
  }
};

} // namespace

static ExegesisTarget *getTheExegesisRISCVTarget() {
  static ExegesisRISCVTarget Target;
  return &Target;
}

void InitializeRISCVExegesisTarget() {
  ExegesisTarget::registerTarget(getTheExegesisRISCVTarget());
}

} // namespace exegesis
} // namespace llvm
