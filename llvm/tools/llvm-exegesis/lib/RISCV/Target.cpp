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
#include "MCTargetDesc/RISCVMCTargetDesc.h"
#include "MCTargetDesc/RISCVMatInt.h"
#include "RISCV.h"
#include "RISCVExegesisPasses.h"
#include "RISCVInstrInfo.h"
#include "RISCVRegisterInfo.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/raw_ostream.h"

// include computeAvailableFeatures and computeRequiredFeatures.
#define GET_AVAILABLE_OPCODE_CHECKER
#include "RISCVGenInstrInfo.inc"

#include "llvm/CodeGen/MachineInstrBuilder.h"

#include <vector>

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

template <class BaseT> class RISCVSnippetGenerator : public BaseT {
  static void printRoundingMode(raw_ostream &OS, unsigned Val, bool UsesVXRM) {
    if (UsesVXRM) {
      assert(RISCVVXRndMode::isValidRoundingMode(Val));
      OS << RISCVVXRndMode::roundingModeToString(
          static_cast<RISCVVXRndMode::RoundingMode>(Val));
    } else {
      assert(RISCVFPRndMode::isValidRoundingMode(Val));
      OS << RISCVFPRndMode::roundingModeToString(
          static_cast<RISCVFPRndMode::RoundingMode>(Val));
    }
  }

  static constexpr unsigned MinSEW = 8;
  // ELEN is basically SEW_max.
  unsigned ELEN = 64;

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
    if (isOpcodeAvailableIn(Opcode, {Feature_HasStdExtZvkshBit}))
      return 256U;
    if (isOpcodeAvailableIn(Opcode, {Feature_HasStdExtZvknhaOrZvknhbBit}))
      // In Zvknh[ab], when SEW=64 is used (i.e. Zvknhb), EGW is 256.
      // Otherwise it's 128.
      return SEW == 64 ? 256U : 128U;

    llvm_unreachable("Unsupported opcode");
  }

  // A handy utility to multiply or divide an integer by LMUL.
  template <typename T> static T multiplyLMul(T Val, RISCVVType::VLMUL VLMul) {
    auto [LMul, IsFractional] = RISCVVType::decodeVLMUL(VLMul);
    return IsFractional ? Val / LMul : Val * LMul;
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
  RISCVSnippetGenerator(const LLVMState &State,
                        const SnippetGenerator::Options &Opts)
      : BaseT(State, Opts),
        AggregateRegisters(State.getRegInfo().getNumRegs(), /*initVal=*/true) {
    // Initialize standalone registers mask.
    const MCRegisterInfo &RegInfo = State.getRegInfo();
    const unsigned StandaloneRegClasses[] = {
        RISCV::GPRRegClassID, RISCV::FPR16RegClassID, RISCV::VRRegClassID};

    for (unsigned RegClassID : StandaloneRegClasses)
      for (unsigned Reg : RegInfo.getRegClass(RegClassID))
        AggregateRegisters.reset(Reg);

    // Initialize ELEN and VLEN.
    // FIXME: We could have obtained these two constants from RISCVSubtarget
    // but in order to get that from TargetMachine, we need a Function.
    const MCSubtargetInfo &STI = State.getSubtargetInfo();
    ELEN = STI.checkFeatures("+zve64x") ? 64 : 32;

    std::string ZvlQuery;
    for (unsigned Size = 32; Size <= 65536; Size *= 2) {
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

static bool isMaskedSibling(unsigned MaskedOp, unsigned UnmaskedOp) {
  const auto *RVVMasked = RISCV::getMaskedPseudoInfo(MaskedOp);
  return RVVMasked && RVVMasked->UnmaskedPseudo == UnmaskedOp;
}

// There are primarily two kinds of opcodes that are not eligible
// in a serial snippet:
// (1) Has a use operand that can not overlap with the def operand
// (i.e. early clobber).
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
  // The permutation instructions listed below cannot have destination
  // overlapping with the source.
  case RISCV::VRGATHEREI16_VV:
  case RISCV::VRGATHER_VI:
  case RISCV::VRGATHER_VV:
  case RISCV::VRGATHER_VX:
  case RISCV::VSLIDE1UP_VX:
  case RISCV::VSLIDEUP_VI:
  case RISCV::VSLIDEUP_VX:
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
void RISCVSnippetGenerator<BaseT>::annotateWithVType(
    const CodeTemplate &OrigCT, const Instruction &Instr, unsigned BaseOpcode,
    const BitVector &ForbiddenRegisters,
    std::vector<CodeTemplate> &Result) const {
  const MCSubtargetInfo &STI = SnippetGenerator::State.getSubtargetInfo();
  unsigned VPseudoOpcode = Instr.getOpcode();

  bool IsSerial = std::is_same_v<BaseT, SerialSnippetGenerator>;

  const MCInstrDesc &MIDesc = Instr.Description;
  const uint64_t TSFlags = MIDesc.TSFlags;

  RISCVVType::VLMUL VLMul = RISCVII::getLMul(TSFlags);

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
    const Operand &SEWOp = Instr.Operands[RISCVII::getSEWOpNum(MIDesc)];
    VTypeOperands.insert(&SEWOp);

    if (SEWOp.Info->OperandType == RISCVOp::OPERAND_SEW_MASK) {
      // If it's a mask-producing instruction, the SEW operand is always zero.
      Log2SEWs.push_back(0);
    } else {
      SmallVector<unsigned, 4> SEWCandidates;

      // (RVV spec 3.4.2) For fractional LMUL, the supported SEW are between
      // [SEW_min, LMUL * ELEN].
      unsigned SEWUpperBound =
          VLMul >= RISCVVType::LMUL_F8 ? multiplyLMul(ELEN, VLMul) : ELEN;
      for (unsigned SEW = MinSEW; SEW <= SEWUpperBound; SEW <<= 1) {
        SEWCandidates.push_back(SEW);

        // Some scheduling classes already integrate SEW; only put
        // their corresponding SEW values at the SEW operands.
        // NOTE: It is imperative to put this condition in the front, otherwise
        // it is tricky and difficult to know if there is an integrated
        // SEW after other rules are applied to filter the candidates.
        const auto *RVVBase =
            RISCVVInversePseudosTable::getBaseInfo(BaseOpcode, VLMul, SEW);
        if (RVVBase && (RVVBase->Pseudo == VPseudoOpcode ||
                        isMaskedSibling(VPseudoOpcode, RVVBase->Pseudo) ||
                        isMaskedSibling(RVVBase->Pseudo, VPseudoOpcode))) {
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

        // The EEW for source operand in VSEXT and VZEXT is a fraction
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
            // Zvknhb supports SEW=64 as well.
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
        Log2SEWs.push_back(Log2_32(SEW));
    }
  } else {
    Log2SEWs.push_back(std::nullopt);
  }

  if (HasPolicyOp) {
    VTypeOperands.insert(&Instr.Operands[RISCVII::getVecPolicyOpNum(MIDesc)]);

    Policies = {0, RISCVVType::TAIL_AGNOSTIC, RISCVVType::MASK_AGNOSTIC,
                (RISCVVType::TAIL_AGNOSTIC | RISCVVType::MASK_AGNOSTIC)};
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

    if (UsesVXRM) {
      // Use RNU as the default VXRM.
      RoundingModes = {RISCVVXRndMode::RNU};
      if (EnumerateRoundingModes)
        RoundingModes.append(
            {RISCVVXRndMode::RNE, RISCVVXRndMode::RDN, RISCVVXRndMode::ROD});
    } else {
      if (EnumerateRoundingModes)
        RoundingModes = {RISCVFPRndMode::RNE, RISCVFPRndMode::RTZ,
                         RISCVFPRndMode::RDN, RISCVFPRndMode::RUP,
                         RISCVFPRndMode::RMM};
      else
        // If we're not enumerating FRM, use DYN to instruct
        // RISCVInsertReadWriteCSRPass to insert nothing.
        RoundingModes = {RISCVFPRndMode::DYN};
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
      SS << LS
         << "Policy: " << (*Policy & RISCVVType::TAIL_AGNOSTIC ? "ta" : "tu")
         << "/" << (*Policy & RISCVVType::MASK_AGNOSTIC ? "ma" : "mu");
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
RISCVSnippetGenerator<BaseT>::generateCodeTemplates(
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

  if (!BaseOpcode)
    return BaseCodeTemplates;

  // Specialize for RVV pseudo.
  std::vector<CodeTemplate> ExpandedTemplates;
  for (const auto &BaseCT : *BaseCodeTemplates)
    annotateWithVType(BaseCT, Instr, BaseOpcode, ForbiddenRegisters,
                      ExpandedTemplates);

  return ExpandedTemplates;
}

// Stores constant value to a general-purpose (integer) register.
static std::vector<MCInst> loadIntReg(const MCSubtargetInfo &STI,
                                      MCRegister Reg, const APInt &Value) {
  SmallVector<MCInst, 8> MCInstSeq;
  MCRegister DestReg = Reg;

  RISCVMatInt::generateMCInstSeq(Value.getSExtValue(), STI, DestReg, MCInstSeq);

  std::vector<MCInst> MatIntInstrs(MCInstSeq.begin(), MCInstSeq.end());
  return MatIntInstrs;
}

const MCPhysReg ScratchIntReg = RISCV::X30; // t5

// Stores constant bits to a floating-point register.
static std::vector<MCInst> loadFPRegBits(const MCSubtargetInfo &STI,
                                         MCRegister Reg, const APInt &Bits,
                                         unsigned FmvOpcode) {
  std::vector<MCInst> Instrs = loadIntReg(STI, ScratchIntReg, Bits);
  Instrs.push_back(MCInstBuilder(FmvOpcode).addReg(Reg).addReg(ScratchIntReg));
  return Instrs;
}

// main idea is:
// we support APInt only if (represented as double) it has zero fractional
// part: 1.0, 2.0, 3.0, etc... then we can do the trick: write int to tmp reg t5
// and then do FCVT this is only reliable thing in 32-bit mode, otherwise we
// need to use __floatsidf
static std::vector<MCInst> loadFP64RegBits32(const MCSubtargetInfo &STI,
                                             MCRegister Reg,
                                             const APInt &Bits) {
  double D = Bits.bitsToDouble();
  double IPart;
  double FPart = std::modf(D, &IPart);

  if (std::abs(FPart) > std::numeric_limits<double>::epsilon()) {
    errs() << "loadFP64RegBits32 is not implemented for doubles like " << D
           << ", please remove fractional part\n";
    return {};
  }

  std::vector<MCInst> Instrs = loadIntReg(STI, ScratchIntReg, Bits);
  Instrs.push_back(
      MCInstBuilder(RISCV::FCVT_D_W).addReg(Reg).addReg(ScratchIntReg));
  return Instrs;
}

class ExegesisRISCVTarget : public ExegesisTarget {
  // NOTE: Alternatively, we can use BitVector here, but the number of RVV MC
  // opcodes is just a small portion of the entire opcode space, so I thought it
  // would be a waste of space to use BitVector.
  mutable SmallSet<unsigned, 16> RVVMCOpcodesWithPseudos;

public:
  ExegesisRISCVTarget();

  bool matchesArch(Triple::ArchType Arch) const override;

  std::vector<MCInst> setRegTo(const MCSubtargetInfo &STI, MCRegister Reg,
                               const APInt &Value) const override;

  const char *getIgnoredOpcodeReasonOrNull(const LLVMState &State,
                                           unsigned Opcode) const override {
    // We don't want to support RVV instructions that depend on VTYPE, because
    // those instructions by themselves don't carry any additional information
    // for us to setup the proper VTYPE environment via VSETVL instructions.
    // FIXME: Ideally, we should use RISCVVInversePseudosTable, but it requires
    // LMUL and SEW and I don't think enumerating those combinations is any
    // better than the ugly trick here that memorizes the corresponding MC
    // opcodes of the RVV pseudo we have processed previously. This works most
    // of the time because RVV pseudo opcodes are placed before any other RVV
    // opcodes. Of course this doesn't work if we're asked to benchmark only a
    // certain subset of opcodes.
    if (RVVMCOpcodesWithPseudos.count(Opcode))
      return "The MC opcode of RVV instructions are ignored";

    // We want to support all RVV pseudos.
    if (unsigned MCOpcode = RISCV::getRVVMCOpcode(Opcode)) {
      RVVMCOpcodesWithPseudos.insert(MCOpcode);
      return nullptr;
    }

    return ExegesisTarget::getIgnoredOpcodeReasonOrNull(State, Opcode);
  }

  MCRegister getDefaultLoopCounterRegister(const Triple &) const override;

  void decrementLoopCounterAndJump(MachineBasicBlock &MBB,
                                   MachineBasicBlock &TargetMBB,
                                   const MCInstrInfo &MII,
                                   MCRegister LoopRegister) const override;

  MCRegister getScratchMemoryRegister(const Triple &TT) const override;

  void fillMemoryOperands(InstructionTemplate &IT, MCRegister Reg,
                          unsigned Offset) const override;

  ArrayRef<MCPhysReg> getUnavailableRegisters() const override;

  bool allowAsBackToBack(const Instruction &Instr) const override {
    return !Instr.Description.isPseudo();
  }

  Error randomizeTargetMCOperand(const Instruction &Instr, const Variable &Var,
                                 MCOperand &AssignedValue,
                                 const BitVector &ForbiddenRegs) const override;

  std::unique_ptr<SnippetGenerator> createSerialSnippetGenerator(
      const LLVMState &State,
      const SnippetGenerator::Options &Opts) const override {
    return std::make_unique<RISCVSnippetGenerator<SerialSnippetGenerator>>(
        State, Opts);
  }

  std::unique_ptr<SnippetGenerator> createParallelSnippetGenerator(
      const LLVMState &State,
      const SnippetGenerator::Options &Opts) const override {
    return std::make_unique<RISCVSnippetGenerator<ParallelSnippetGenerator>>(
        State, Opts);
  }

  std::vector<InstructionTemplate>
  generateInstructionVariants(const Instruction &Instr,
                              unsigned MaxConfigsPerOpcode) const override;

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

ExegesisRISCVTarget::ExegesisRISCVTarget()
    : ExegesisTarget(RISCVCpuPfmCounters, RISCV_MC::isOpcodeAvailable) {}

bool ExegesisRISCVTarget::matchesArch(Triple::ArchType Arch) const {
  return Arch == Triple::riscv32 || Arch == Triple::riscv64;
}

std::vector<MCInst> ExegesisRISCVTarget::setRegTo(const MCSubtargetInfo &STI,
                                                  MCRegister Reg,
                                                  const APInt &Value) const {
  if (RISCV::GPRRegClass.contains(Reg))
    return loadIntReg(STI, Reg, Value);
  if (RISCV::FPR16RegClass.contains(Reg))
    return loadFPRegBits(STI, Reg, Value, RISCV::FMV_H_X);
  if (RISCV::FPR32RegClass.contains(Reg))
    return loadFPRegBits(STI, Reg, Value, RISCV::FMV_W_X);
  if (RISCV::FPR64RegClass.contains(Reg)) {
    if (STI.hasFeature(RISCV::Feature64Bit))
      return loadFPRegBits(STI, Reg, Value, RISCV::FMV_D_X);
    return loadFP64RegBits32(STI, Reg, Value);
  }
  // TODO: Emit proper code to initialize other kinds of registers.
  return {};
}

const MCPhysReg DefaultLoopCounterReg = RISCV::X31; // t6
const MCPhysReg ScratchMemoryReg = RISCV::X10;      // a0

MCRegister
ExegesisRISCVTarget::getDefaultLoopCounterRegister(const Triple &) const {
  return DefaultLoopCounterReg;
}

void ExegesisRISCVTarget::decrementLoopCounterAndJump(
    MachineBasicBlock &MBB, MachineBasicBlock &TargetMBB,
    const MCInstrInfo &MII, MCRegister LoopRegister) const {
  BuildMI(&MBB, DebugLoc(), MII.get(RISCV::ADDI))
      .addDef(LoopRegister)
      .addUse(LoopRegister)
      .addImm(-1);
  BuildMI(&MBB, DebugLoc(), MII.get(RISCV::BNE))
      .addUse(LoopRegister)
      .addUse(RISCV::X0)
      .addMBB(&TargetMBB);
}

MCRegister
ExegesisRISCVTarget::getScratchMemoryRegister(const Triple &TT) const {
  return ScratchMemoryReg; // a0
}

void ExegesisRISCVTarget::fillMemoryOperands(InstructionTemplate &IT,
                                             MCRegister Reg,
                                             unsigned Offset) const {
  // TODO: for now we ignore Offset because have no way
  // to detect it in instruction.
  auto &I = IT.getInstr();

  auto MemOpIt =
      find_if(I.Operands, [](const Operand &Op) { return Op.isMemory(); });
  assert(MemOpIt != I.Operands.end() &&
         "Instruction must have memory operands");

  const Operand &MemOp = *MemOpIt;

  assert(MemOp.isReg() && "Memory operand expected to be register");

  IT.getValueFor(MemOp) = MCOperand::createReg(Reg);
}

const MCPhysReg UnavailableRegisters[4] = {RISCV::X0, DefaultLoopCounterReg,
                                           ScratchIntReg, ScratchMemoryReg};

ArrayRef<MCPhysReg> ExegesisRISCVTarget::getUnavailableRegisters() const {
  return UnavailableRegisters;
}

Error ExegesisRISCVTarget::randomizeTargetMCOperand(
    const Instruction &Instr, const Variable &Var, MCOperand &AssignedValue,
    const BitVector &ForbiddenRegs) const {
  uint8_t OperandType =
      Instr.getPrimaryOperand(Var).getExplicitOperandInfo().OperandType;

  switch (OperandType) {
  case RISCVOp::OPERAND_FRMARG:
    AssignedValue = MCOperand::createImm(RISCVFPRndMode::DYN);
    break;
  case RISCVOp::OPERAND_SIMM10_LSB0000_NONZERO:
    AssignedValue = MCOperand::createImm(0b1 << 4);
    break;
  case RISCVOp::OPERAND_SIMM6_NONZERO:
  case RISCVOp::OPERAND_UIMMLOG2XLEN_NONZERO:
    AssignedValue = MCOperand::createImm(1);
    break;
  case RISCVOp::OPERAND_SIMM5:
    // 5-bit signed immediate value.
    AssignedValue = MCOperand::createImm(randomIndex(31) - 16);
    break;
  case RISCVOp::OPERAND_AVL:
  case RISCVOp::OPERAND_UIMM5:
    // 5-bit unsigned immediate value.
    AssignedValue = MCOperand::createImm(randomIndex(31));
    break;
  default:
    if (OperandType >= RISCVOp::OPERAND_FIRST_RISCV_IMM &&
        OperandType <= RISCVOp::OPERAND_LAST_RISCV_IMM)
      AssignedValue = MCOperand::createImm(0);
  }
  return Error::success();
}

std::vector<InstructionTemplate>
ExegesisRISCVTarget::generateInstructionVariants(
    const Instruction &Instr, unsigned int MaxConfigsPerOpcode) const {
  InstructionTemplate IT{&Instr};
  for (const Operand &Op : Instr.Operands)
    if (Op.isMemory()) {
      IT.getValueFor(Op) = MCOperand::createReg(ScratchMemoryReg);
    }
  return {IT};
}

} // anonymous namespace

static ExegesisTarget *getTheRISCVExegesisTarget() {
  static ExegesisRISCVTarget Target;
  return &Target;
}

void InitializeRISCVExegesisTarget() {
  ExegesisTarget::registerTarget(getTheRISCVExegesisTarget());
}

} // namespace exegesis
} // namespace llvm
