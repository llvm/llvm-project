//===-- Target.cpp ----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "../Target.h"

#include "../Latency.h"
#include "../SnippetGenerator.h"
#include "../Uops.h"
#include "MCTargetDesc/X86BaseInfo.h"
#include "MCTargetDesc/X86MCTargetDesc.h"
#include "X86.h"
#include "X86RegisterInfo.h"
#include "X86Subtarget.h"
#include "llvm/MC/MCInstBuilder.h"

namespace llvm {
namespace exegesis {

// Returns an error if we cannot handle the memory references in this
// instruction.
static Error isInvalidMemoryInstr(const Instruction &Instr) {
  switch (Instr.Description->TSFlags & X86II::FormMask) {
  default:
    llvm_unreachable("Unknown FormMask value");
  // These have no memory access.
  case X86II::Pseudo:
  case X86II::RawFrm:
  case X86II::AddCCFrm:
  case X86II::MRMDestReg:
  case X86II::MRMSrcReg:
  case X86II::MRMSrcReg4VOp3:
  case X86II::MRMSrcRegOp4:
  case X86II::MRMSrcRegCC:
  case X86II::MRMXrCC:
  case X86II::MRMXr:
  case X86II::MRM0r:
  case X86II::MRM1r:
  case X86II::MRM2r:
  case X86II::MRM3r:
  case X86II::MRM4r:
  case X86II::MRM5r:
  case X86II::MRM6r:
  case X86II::MRM7r:
  case X86II::MRM_C0:
  case X86II::MRM_C1:
  case X86II::MRM_C2:
  case X86II::MRM_C3:
  case X86II::MRM_C4:
  case X86II::MRM_C5:
  case X86II::MRM_C6:
  case X86II::MRM_C7:
  case X86II::MRM_C8:
  case X86II::MRM_C9:
  case X86II::MRM_CA:
  case X86II::MRM_CB:
  case X86II::MRM_CC:
  case X86II::MRM_CD:
  case X86II::MRM_CE:
  case X86II::MRM_CF:
  case X86II::MRM_D0:
  case X86II::MRM_D1:
  case X86II::MRM_D2:
  case X86II::MRM_D3:
  case X86II::MRM_D4:
  case X86II::MRM_D5:
  case X86II::MRM_D6:
  case X86II::MRM_D7:
  case X86II::MRM_D8:
  case X86II::MRM_D9:
  case X86II::MRM_DA:
  case X86II::MRM_DB:
  case X86II::MRM_DC:
  case X86II::MRM_DD:
  case X86II::MRM_DE:
  case X86II::MRM_DF:
  case X86II::MRM_E0:
  case X86II::MRM_E1:
  case X86II::MRM_E2:
  case X86II::MRM_E3:
  case X86II::MRM_E4:
  case X86II::MRM_E5:
  case X86II::MRM_E6:
  case X86II::MRM_E7:
  case X86II::MRM_E8:
  case X86II::MRM_E9:
  case X86II::MRM_EA:
  case X86II::MRM_EB:
  case X86II::MRM_EC:
  case X86II::MRM_ED:
  case X86II::MRM_EE:
  case X86II::MRM_EF:
  case X86II::MRM_F0:
  case X86II::MRM_F1:
  case X86II::MRM_F2:
  case X86II::MRM_F3:
  case X86II::MRM_F4:
  case X86II::MRM_F5:
  case X86II::MRM_F6:
  case X86II::MRM_F7:
  case X86II::MRM_F8:
  case X86II::MRM_F9:
  case X86II::MRM_FA:
  case X86II::MRM_FB:
  case X86II::MRM_FC:
  case X86II::MRM_FD:
  case X86II::MRM_FE:
  case X86II::MRM_FF:
  case X86II::RawFrmImm8:
    return Error::success();
  case X86II::AddRegFrm:
    return (Instr.Description->Opcode == X86::POP16r || Instr.Description->Opcode == X86::POP32r ||
            Instr.Description->Opcode == X86::PUSH16r || Instr.Description->Opcode == X86::PUSH32r)
               ? make_error<BenchmarkFailure>(
                     "unsupported opcode: unsupported memory access")
               : Error::success();
  // These access memory and are handled.
  case X86II::MRMDestMem:
  case X86II::MRMSrcMem:
  case X86II::MRMSrcMem4VOp3:
  case X86II::MRMSrcMemOp4:
  case X86II::MRMSrcMemCC:
  case X86II::MRMXmCC:
  case X86II::MRMXm:
  case X86II::MRM0m:
  case X86II::MRM1m:
  case X86II::MRM2m:
  case X86II::MRM3m:
  case X86II::MRM4m:
  case X86II::MRM5m:
  case X86II::MRM6m:
  case X86II::MRM7m:
    return Error::success();
  // These access memory and are not handled yet.
  case X86II::RawFrmImm16:
  case X86II::RawFrmMemOffs:
  case X86II::RawFrmSrc:
  case X86II::RawFrmDst:
  case X86II::RawFrmDstSrc:
    return make_error<BenchmarkFailure>(
        "unsupported opcode: non uniform memory access");
  }
}

static llvm::Error IsInvalidOpcode(const Instruction &Instr) {
  const auto OpcodeName = Instr.Name;
  if ((Instr.Description->TSFlags & X86II::FormMask) == X86II::Pseudo)
    return llvm::make_error<BenchmarkFailure>(
        "unsupported opcode: pseudo instruction");
  if (OpcodeName.startswith("POPF") || OpcodeName.startswith("PUSHF") ||
      OpcodeName.startswith("ADJCALLSTACK"))
    return llvm::make_error<BenchmarkFailure>(
        "unsupported opcode: Push/Pop/AdjCallStack");
  if (llvm::Error Error = isInvalidMemoryInstr(Instr))
    return Error;
  // We do not handle instructions with OPERAND_PCREL.
  for (const Operand &Op : Instr.Operands)
    if (Op.isExplicit() &&
        Op.getExplicitOperandInfo().OperandType == llvm::MCOI::OPERAND_PCREL)
      return llvm::make_error<BenchmarkFailure>(
          "unsupported opcode: PC relative operand");
  // We do not handle second-form X87 instructions. We only handle first-form
  // ones (_Fp), see comment in X86InstrFPStack.td.
  for (const Operand &Op : Instr.Operands)
    if (Op.isReg() && Op.isExplicit() &&
        Op.getExplicitOperandInfo().RegClass == llvm::X86::RSTRegClassID)
      return llvm::make_error<BenchmarkFailure>(
          "unsupported second-form X87 instruction");
  return llvm::Error::success();
}

static unsigned getX86FPFlags(const Instruction &Instr) {
  return Instr.Description->TSFlags & llvm::X86II::FPTypeMask;
}

namespace {
class X86LatencySnippetGenerator : public LatencySnippetGenerator {
public:
  using LatencySnippetGenerator::LatencySnippetGenerator;

  llvm::Expected<std::vector<CodeTemplate>>
  generateCodeTemplates(const Instruction &Instr) const override;
};
} // namespace

llvm::Expected<std::vector<CodeTemplate>>
X86LatencySnippetGenerator::generateCodeTemplates(
    const Instruction &Instr) const {
  if (auto E = IsInvalidOpcode(Instr))
    return std::move(E);

  switch (getX86FPFlags(Instr)) {
  case llvm::X86II::NotFP:
    return LatencySnippetGenerator::generateCodeTemplates(Instr);
  case llvm::X86II::ZeroArgFP:
  case llvm::X86II::OneArgFP:
  case llvm::X86II::SpecialFP:
  case llvm::X86II::CompareFP:
  case llvm::X86II::CondMovFP:
    return llvm::make_error<BenchmarkFailure>("Unsupported x87 Instruction");
  case llvm::X86II::OneArgFPRW:
  case llvm::X86II::TwoArgFP:
    // These are instructions like
    //   - `ST(0) = fsqrt(ST(0))` (OneArgFPRW)
    //   - `ST(0) = ST(0) + ST(i)` (TwoArgFP)
    // They are intrinsically serial and do not modify the state of the stack.
    return generateSelfAliasingCodeTemplates(Instr);
  default:
    llvm_unreachable("Unknown FP Type!");
  }
}

namespace {
class X86UopsSnippetGenerator : public UopsSnippetGenerator {
public:
  using UopsSnippetGenerator::UopsSnippetGenerator;

  llvm::Expected<std::vector<CodeTemplate>>
  generateCodeTemplates(const Instruction &Instr) const override;
};
} // namespace

llvm::Expected<std::vector<CodeTemplate>>
X86UopsSnippetGenerator::generateCodeTemplates(
    const Instruction &Instr) const {
  if (auto E = IsInvalidOpcode(Instr))
    return std::move(E);

  switch (getX86FPFlags(Instr)) {
  case llvm::X86II::NotFP:
    return UopsSnippetGenerator::generateCodeTemplates(Instr);
  case llvm::X86II::ZeroArgFP:
  case llvm::X86II::OneArgFP:
  case llvm::X86II::SpecialFP:
    return llvm::make_error<BenchmarkFailure>("Unsupported x87 Instruction");
  case llvm::X86II::OneArgFPRW:
  case llvm::X86II::TwoArgFP:
    // These are instructions like
    //   - `ST(0) = fsqrt(ST(0))` (OneArgFPRW)
    //   - `ST(0) = ST(0) + ST(i)` (TwoArgFP)
    // They are intrinsically serial and do not modify the state of the stack.
    // We generate the same code for latency and uops.
    return generateSelfAliasingCodeTemplates(Instr);
  case llvm::X86II::CompareFP:
  case llvm::X86II::CondMovFP:
    // We can compute uops for any FP instruction that does not grow or shrink
    // the stack (either do not touch the stack or push as much as they pop).
    return generateUnconstrainedCodeTemplates(
        Instr, "instruction does not grow/shrink the FP stack");
  default:
    llvm_unreachable("Unknown FP Type!");
  }
}

static unsigned getLoadImmediateOpcode(unsigned RegBitWidth) {
  switch (RegBitWidth) {
  case 8:
    return llvm::X86::MOV8ri;
  case 16:
    return llvm::X86::MOV16ri;
  case 32:
    return llvm::X86::MOV32ri;
  case 64:
    return llvm::X86::MOV64ri;
  }
  llvm_unreachable("Invalid Value Width");
}

// Generates instruction to load an immediate value into a register.
static llvm::MCInst loadImmediate(unsigned Reg, unsigned RegBitWidth,
                                  const llvm::APInt &Value) {
  if (Value.getBitWidth() > RegBitWidth)
    llvm_unreachable("Value must fit in the Register");
  return llvm::MCInstBuilder(getLoadImmediateOpcode(RegBitWidth))
      .addReg(Reg)
      .addImm(Value.getZExtValue());
}

// Allocates scratch memory on the stack.
static llvm::MCInst allocateStackSpace(unsigned Bytes) {
  return llvm::MCInstBuilder(llvm::X86::SUB64ri8)
      .addReg(llvm::X86::RSP)
      .addReg(llvm::X86::RSP)
      .addImm(Bytes);
}

// Fills scratch memory at offset `OffsetBytes` with value `Imm`.
static llvm::MCInst fillStackSpace(unsigned MovOpcode, unsigned OffsetBytes,
                                   uint64_t Imm) {
  return llvm::MCInstBuilder(MovOpcode)
      // Address = ESP
      .addReg(llvm::X86::RSP) // BaseReg
      .addImm(1)              // ScaleAmt
      .addReg(0)              // IndexReg
      .addImm(OffsetBytes)    // Disp
      .addReg(0)              // Segment
      // Immediate.
      .addImm(Imm);
}

// Loads scratch memory into register `Reg` using opcode `RMOpcode`.
static llvm::MCInst loadToReg(unsigned Reg, unsigned RMOpcode) {
  return llvm::MCInstBuilder(RMOpcode)
      .addReg(Reg)
      // Address = ESP
      .addReg(llvm::X86::RSP) // BaseReg
      .addImm(1)              // ScaleAmt
      .addReg(0)              // IndexReg
      .addImm(0)              // Disp
      .addReg(0);             // Segment
}

// Releases scratch memory.
static llvm::MCInst releaseStackSpace(unsigned Bytes) {
  return llvm::MCInstBuilder(llvm::X86::ADD64ri8)
      .addReg(llvm::X86::RSP)
      .addReg(llvm::X86::RSP)
      .addImm(Bytes);
}

// Reserves some space on the stack, fills it with the content of the provided
// constant and provide methods to load the stack value into a register.
namespace {
struct ConstantInliner {
  explicit ConstantInliner(const llvm::APInt &Constant) : Constant_(Constant) {}

  std::vector<llvm::MCInst> loadAndFinalize(unsigned Reg, unsigned RegBitWidth,
                                            unsigned Opcode);

  std::vector<llvm::MCInst> loadX87STAndFinalize(unsigned Reg);

  std::vector<llvm::MCInst> loadX87FPAndFinalize(unsigned Reg);

  std::vector<llvm::MCInst> popFlagAndFinalize();

private:
  ConstantInliner &add(const llvm::MCInst &Inst) {
    Instructions.push_back(Inst);
    return *this;
  }

  void initStack(unsigned Bytes);

  static constexpr const unsigned kF80Bytes = 10; // 80 bits.

  llvm::APInt Constant_;
  std::vector<llvm::MCInst> Instructions;
};
} // namespace

std::vector<llvm::MCInst> ConstantInliner::loadAndFinalize(unsigned Reg,
                                                           unsigned RegBitWidth,
                                                           unsigned Opcode) {
  assert((RegBitWidth & 7) == 0 && "RegBitWidth must be a multiple of 8 bits");
  initStack(RegBitWidth / 8);
  add(loadToReg(Reg, Opcode));
  add(releaseStackSpace(RegBitWidth / 8));
  return std::move(Instructions);
}

std::vector<llvm::MCInst> ConstantInliner::loadX87STAndFinalize(unsigned Reg) {
  initStack(kF80Bytes);
  add(llvm::MCInstBuilder(llvm::X86::LD_F80m)
          // Address = ESP
          .addReg(llvm::X86::RSP) // BaseReg
          .addImm(1)              // ScaleAmt
          .addReg(0)              // IndexReg
          .addImm(0)              // Disp
          .addReg(0));            // Segment
  if (Reg != llvm::X86::ST0)
    add(llvm::MCInstBuilder(llvm::X86::ST_Frr).addReg(Reg));
  add(releaseStackSpace(kF80Bytes));
  return std::move(Instructions);
}

std::vector<llvm::MCInst> ConstantInliner::loadX87FPAndFinalize(unsigned Reg) {
  initStack(kF80Bytes);
  add(llvm::MCInstBuilder(llvm::X86::LD_Fp80m)
          .addReg(Reg)
          // Address = ESP
          .addReg(llvm::X86::RSP) // BaseReg
          .addImm(1)              // ScaleAmt
          .addReg(0)              // IndexReg
          .addImm(0)              // Disp
          .addReg(0));            // Segment
  add(releaseStackSpace(kF80Bytes));
  return std::move(Instructions);
}

std::vector<llvm::MCInst> ConstantInliner::popFlagAndFinalize() {
  initStack(8);
  add(llvm::MCInstBuilder(llvm::X86::POPF64));
  return std::move(Instructions);
}

void ConstantInliner::initStack(unsigned Bytes) {
  assert(Constant_.getBitWidth() <= Bytes * 8 &&
         "Value does not have the correct size");
  const llvm::APInt WideConstant = Constant_.getBitWidth() < Bytes * 8
                                       ? Constant_.sext(Bytes * 8)
                                       : Constant_;
  add(allocateStackSpace(Bytes));
  size_t ByteOffset = 0;
  for (; Bytes - ByteOffset >= 4; ByteOffset += 4)
    add(fillStackSpace(
        llvm::X86::MOV32mi, ByteOffset,
        WideConstant.extractBits(32, ByteOffset * 8).getZExtValue()));
  if (Bytes - ByteOffset >= 2) {
    add(fillStackSpace(
        llvm::X86::MOV16mi, ByteOffset,
        WideConstant.extractBits(16, ByteOffset * 8).getZExtValue()));
    ByteOffset += 2;
  }
  if (Bytes - ByteOffset >= 1)
    add(fillStackSpace(
        llvm::X86::MOV8mi, ByteOffset,
        WideConstant.extractBits(8, ByteOffset * 8).getZExtValue()));
}

#include "X86GenExegesis.inc"

namespace {
class ExegesisX86Target : public ExegesisTarget {
public:
  ExegesisX86Target() : ExegesisTarget(X86CpuPfmCounters) {}

private:
  void addTargetSpecificPasses(llvm::PassManagerBase &PM) const override;

  unsigned getScratchMemoryRegister(const llvm::Triple &TT) const override;

  unsigned getMaxMemoryAccessSize() const override { return 64; }

  void randomizeMCOperand(const Instruction &Instr, const Variable &Var,
                          llvm::MCOperand &AssignedValue,
                          const llvm::BitVector &ForbiddenRegs) const override;

  void fillMemoryOperands(InstructionTemplate &IT, unsigned Reg,
                          unsigned Offset) const override;

  std::vector<llvm::MCInst> setRegTo(const llvm::MCSubtargetInfo &STI,
                                     unsigned Reg,
                                     const llvm::APInt &Value) const override;

  ArrayRef<unsigned> getUnavailableRegisters() const override {
    return makeArrayRef(kUnavailableRegisters,
                        sizeof(kUnavailableRegisters) /
                            sizeof(kUnavailableRegisters[0]));
  }

  std::unique_ptr<SnippetGenerator>
  createLatencySnippetGenerator(const LLVMState &State) const override {
    return llvm::make_unique<X86LatencySnippetGenerator>(State);
  }

  std::unique_ptr<SnippetGenerator>
  createUopsSnippetGenerator(const LLVMState &State) const override {
    return llvm::make_unique<X86UopsSnippetGenerator>(State);
  }

  bool matchesArch(llvm::Triple::ArchType Arch) const override {
    return Arch == llvm::Triple::x86_64 || Arch == llvm::Triple::x86;
  }

  static const unsigned kUnavailableRegisters[4];
};

// We disable a few registers that cannot be encoded on instructions with a REX
// prefix.
const unsigned ExegesisX86Target::kUnavailableRegisters[4] = {X86::AH, X86::BH,
                                                              X86::CH, X86::DH};
} // namespace

void ExegesisX86Target::addTargetSpecificPasses(
    llvm::PassManagerBase &PM) const {
  // Lowers FP pseudo-instructions, e.g. ABS_Fp32 -> ABS_F.
  PM.add(llvm::createX86FloatingPointStackifierPass());
}

unsigned
ExegesisX86Target::getScratchMemoryRegister(const llvm::Triple &TT) const {
  if (!TT.isArch64Bit()) {
    // FIXME: This would require popping from the stack, so we would have to
    // add some additional setup code.
    return 0;
  }
  return TT.isOSWindows() ? llvm::X86::RCX : llvm::X86::RDI;
}

void ExegesisX86Target::randomizeMCOperand(
    const Instruction &Instr, const Variable &Var,
    llvm::MCOperand &AssignedValue,
    const llvm::BitVector &ForbiddenRegs) const {
  ExegesisTarget::randomizeMCOperand(Instr, Var, AssignedValue, ForbiddenRegs);

  const Operand &Op = Instr.getPrimaryOperand(Var);
  switch (Op.getExplicitOperandInfo().OperandType) {
  case llvm::X86::OperandType::OPERAND_COND_CODE:
    AssignedValue = llvm::MCOperand::createImm(
        randomIndex(llvm::X86::CondCode::LAST_VALID_COND));
    break;
  default:
    break;
  }
}

void ExegesisX86Target::fillMemoryOperands(InstructionTemplate &IT,
                                           unsigned Reg,
                                           unsigned Offset) const {
  assert(!isInvalidMemoryInstr(IT.Instr) &&
         "fillMemoryOperands requires a valid memory instruction");
  int MemOpIdx = X86II::getMemoryOperandNo(IT.Instr.Description->TSFlags);
  assert(MemOpIdx >= 0 && "invalid memory operand index");
  // getMemoryOperandNo() ignores tied operands, so we have to add them back.
  for (unsigned I = 0; I <= static_cast<unsigned>(MemOpIdx); ++I) {
    const auto &Op = IT.Instr.Operands[I];
    if (Op.isTied() && Op.getTiedToIndex() < I) {
      ++MemOpIdx;
    }
  }
  // Now fill in the memory operands.
  const auto SetOp = [&IT](int OpIdx, const MCOperand &OpVal) {
    const auto Op = IT.Instr.Operands[OpIdx];
    assert(Op.isMemory() && Op.isExplicit() && "invalid memory pattern");
    IT.getValueFor(Op) = OpVal;
  };
  SetOp(MemOpIdx + 0, MCOperand::createReg(Reg));    // BaseReg
  SetOp(MemOpIdx + 1, MCOperand::createImm(1));      // ScaleAmt
  SetOp(MemOpIdx + 2, MCOperand::createReg(0));      // IndexReg
  SetOp(MemOpIdx + 3, MCOperand::createImm(Offset)); // Disp
  SetOp(MemOpIdx + 4, MCOperand::createReg(0));      // Segment
}

std::vector<llvm::MCInst>
ExegesisX86Target::setRegTo(const llvm::MCSubtargetInfo &STI, unsigned Reg,
                            const llvm::APInt &Value) const {
  if (llvm::X86::GR8RegClass.contains(Reg))
    return {loadImmediate(Reg, 8, Value)};
  if (llvm::X86::GR16RegClass.contains(Reg))
    return {loadImmediate(Reg, 16, Value)};
  if (llvm::X86::GR32RegClass.contains(Reg))
    return {loadImmediate(Reg, 32, Value)};
  if (llvm::X86::GR64RegClass.contains(Reg))
    return {loadImmediate(Reg, 64, Value)};
  ConstantInliner CI(Value);
  if (llvm::X86::VR64RegClass.contains(Reg))
    return CI.loadAndFinalize(Reg, 64, llvm::X86::MMX_MOVQ64rm);
  if (llvm::X86::VR128XRegClass.contains(Reg)) {
    if (STI.getFeatureBits()[llvm::X86::FeatureAVX512])
      return CI.loadAndFinalize(Reg, 128, llvm::X86::VMOVDQU32Z128rm);
    if (STI.getFeatureBits()[llvm::X86::FeatureAVX])
      return CI.loadAndFinalize(Reg, 128, llvm::X86::VMOVDQUrm);
    return CI.loadAndFinalize(Reg, 128, llvm::X86::MOVDQUrm);
  }
  if (llvm::X86::VR256XRegClass.contains(Reg)) {
    if (STI.getFeatureBits()[llvm::X86::FeatureAVX512])
      return CI.loadAndFinalize(Reg, 256, llvm::X86::VMOVDQU32Z256rm);
    if (STI.getFeatureBits()[llvm::X86::FeatureAVX])
      return CI.loadAndFinalize(Reg, 256, llvm::X86::VMOVDQUYrm);
  }
  if (llvm::X86::VR512RegClass.contains(Reg))
    if (STI.getFeatureBits()[llvm::X86::FeatureAVX512])
      return CI.loadAndFinalize(Reg, 512, llvm::X86::VMOVDQU32Zrm);
  if (llvm::X86::RSTRegClass.contains(Reg)) {
    return CI.loadX87STAndFinalize(Reg);
  }
  if (llvm::X86::RFP32RegClass.contains(Reg) ||
      llvm::X86::RFP64RegClass.contains(Reg) ||
      llvm::X86::RFP80RegClass.contains(Reg)) {
    return CI.loadX87FPAndFinalize(Reg);
  }
  if (Reg == llvm::X86::EFLAGS)
    return CI.popFlagAndFinalize();
  return {}; // Not yet implemented.
}

static ExegesisTarget *getTheExegesisX86Target() {
  static ExegesisX86Target Target;
  return &Target;
}

void InitializeX86ExegesisTarget() {
  ExegesisTarget::registerTarget(getTheExegesisX86Target());
}

} // namespace exegesis
} // namespace llvm
