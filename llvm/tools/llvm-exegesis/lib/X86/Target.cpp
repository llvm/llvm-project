//===-- Target.cpp ----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#include "../Target.h"

#include "../Latency.h"
#include "../Uops.h"
#include "MCTargetDesc/X86BaseInfo.h"
#include "MCTargetDesc/X86MCTargetDesc.h"
#include "X86.h"
#include "X86RegisterInfo.h"
#include "X86Subtarget.h"
#include "llvm/MC/MCInstBuilder.h"

namespace llvm {
namespace exegesis {

namespace {

// A chunk of instruction's operands that represents a single memory access.
struct MemoryOperandRange {
  MemoryOperandRange(llvm::ArrayRef<Operand> Operands) : Ops(Operands) {}

  // Setup InstructionTemplate so the memory access represented by this object
  // points to [reg] + offset.
  void fillOrDie(InstructionTemplate &IT, unsigned Reg, unsigned Offset) {
    switch (Ops.size()) {
    case 5:
      IT.getValueFor(Ops[0]) = llvm::MCOperand::createReg(Reg);    // BaseReg
      IT.getValueFor(Ops[1]) = llvm::MCOperand::createImm(1);      // ScaleAmt
      IT.getValueFor(Ops[2]) = llvm::MCOperand::createReg(0);      // IndexReg
      IT.getValueFor(Ops[3]) = llvm::MCOperand::createImm(Offset); // Disp
      IT.getValueFor(Ops[4]) = llvm::MCOperand::createReg(0);      // Segment
      break;
    default:
      llvm::errs() << Ops.size() << "-op are not handled right now ("
                   << IT.Instr.Name << ")\n";
      llvm_unreachable("Invalid memory configuration");
    }
  }

  // Returns whether Range can be filled.
  static bool isValid(const MemoryOperandRange &Range) {
    return Range.Ops.size() == 5;
  }

  // Returns whether Op is a valid memory operand.
  static bool isMemoryOperand(const Operand &Op) {
    return Op.isMemory() && Op.isExplicit();
  }

  llvm::ArrayRef<Operand> Ops;
};

// X86 memory access involve non constant number of operands, this function
// extracts contiguous memory operands into MemoryOperandRange so it's easier to
// check and fill.
static std::vector<MemoryOperandRange>
getMemoryOperandRanges(llvm::ArrayRef<Operand> Operands) {
  std::vector<MemoryOperandRange> Result;
  while (!Operands.empty()) {
    Operands = Operands.drop_until(MemoryOperandRange::isMemoryOperand);
    auto MemoryOps = Operands.take_while(MemoryOperandRange::isMemoryOperand);
    if (!MemoryOps.empty())
      Result.push_back(MemoryOps);
    Operands = Operands.drop_front(MemoryOps.size());
  }
  return Result;
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
  const bool ValidMemoryOperands = llvm::all_of(
      getMemoryOperandRanges(Instr.Operands), MemoryOperandRange::isValid);
  if (!ValidMemoryOperands)
    return llvm::make_error<BenchmarkFailure>(
        "unsupported opcode: non uniform memory access");
  // We do not handle instructions with OPERAND_PCREL.
  for (const Operand &Op : Instr.Operands)
    if (Op.isExplicit() &&
        Op.getExplicitOperandInfo().OperandType == llvm::MCOI::OPERAND_PCREL)
      return llvm::make_error<BenchmarkFailure>(
          "unsupported opcode: PC relative operand");
  for (const Operand &Op : Instr.Operands)
    if (Op.isReg() && Op.isExplicit() &&
        Op.getExplicitOperandInfo().RegClass ==
            llvm::X86::SEGMENT_REGRegClassID)
      return llvm::make_error<BenchmarkFailure>(
          "unsupported opcode: access segment memory");
  // We do not handle second-form X87 instructions. We only handle first-form
  // ones (_Fp), see comment in X86InstrFPStack.td.
  for (const Operand &Op : Instr.Operands)
    if (Op.isReg() && Op.isExplicit() &&
        Op.getExplicitOperandInfo().RegClass == llvm::X86::RSTRegClassID)
      return llvm::make_error<BenchmarkFailure>(
          "unsupported second-form X87 instruction");
  return llvm::Error::success();
}

static unsigned GetX86FPFlags(const Instruction &Instr) {
  return Instr.Description->TSFlags & llvm::X86II::FPTypeMask;
}

class X86LatencySnippetGenerator : public LatencySnippetGenerator {
public:
  using LatencySnippetGenerator::LatencySnippetGenerator;

  llvm::Expected<std::vector<CodeTemplate>>
  generateCodeTemplates(const Instruction &Instr) const override {
    if (auto E = IsInvalidOpcode(Instr))
      return std::move(E);

    switch (GetX86FPFlags(Instr)) {
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
};

class X86UopsSnippetGenerator : public UopsSnippetGenerator {
public:
  using UopsSnippetGenerator::UopsSnippetGenerator;

  llvm::Expected<std::vector<CodeTemplate>>
  generateCodeTemplates(const Instruction &Instr) const override {
    if (auto E = IsInvalidOpcode(Instr))
      return std::move(E);

    switch (GetX86FPFlags(Instr)) {
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
};

static unsigned GetLoadImmediateOpcode(unsigned RegBitWidth) {
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
  return llvm::MCInstBuilder(GetLoadImmediateOpcode(RegBitWidth))
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
struct ConstantInliner {
  explicit ConstantInliner(const llvm::APInt &Constant) : Constant_(Constant) {}

  std::vector<llvm::MCInst> loadAndFinalize(unsigned Reg, unsigned RegBitWidth,
                                            unsigned Opcode) {
    assert((RegBitWidth & 7) == 0 &&
           "RegBitWidth must be a multiple of 8 bits");
    initStack(RegBitWidth / 8);
    add(loadToReg(Reg, Opcode));
    add(releaseStackSpace(RegBitWidth / 8));
    return std::move(Instructions);
  }

  std::vector<llvm::MCInst> loadX87STAndFinalize(unsigned Reg) {
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

  std::vector<llvm::MCInst> loadX87FPAndFinalize(unsigned Reg) {
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

  std::vector<llvm::MCInst> popFlagAndFinalize() {
    initStack(8);
    add(llvm::MCInstBuilder(llvm::X86::POPF64));
    return std::move(Instructions);
  }

private:
  static constexpr const unsigned kF80Bytes = 10; // 80 bits.

  ConstantInliner &add(const llvm::MCInst &Inst) {
    Instructions.push_back(Inst);
    return *this;
  }

  void initStack(unsigned Bytes) {
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

  llvm::APInt Constant_;
  std::vector<llvm::MCInst> Instructions;
};

#include "X86GenExegesis.inc"

class ExegesisX86Target : public ExegesisTarget {
public:
  ExegesisX86Target() : ExegesisTarget(X86CpuPfmCounters) {}

private:
  void addTargetSpecificPasses(llvm::PassManagerBase &PM) const override {
    // Lowers FP pseudo-instructions, e.g. ABS_Fp32 -> ABS_F.
    PM.add(llvm::createX86FloatingPointStackifierPass());
  }

  unsigned getScratchMemoryRegister(const llvm::Triple &TT) const override {
    if (!TT.isArch64Bit()) {
      // FIXME: This would require popping from the stack, so we would have to
      // add some additional setup code.
      return 0;
    }
    return TT.isOSWindows() ? llvm::X86::RCX : llvm::X86::RDI;
  }

  unsigned getMaxMemoryAccessSize() const override { return 64; }

  void fillMemoryOperands(InstructionTemplate &IT, unsigned Reg,
                          unsigned Offset) const override {
    // FIXME: For instructions that read AND write to memory, we use the same
    // value for input and output.
    for (auto &MemoryRange : getMemoryOperandRanges(IT.Instr.Operands))
      MemoryRange.fillOrDie(IT, Reg, Offset);
  }

  std::vector<llvm::MCInst> setRegTo(const llvm::MCSubtargetInfo &STI,
                                     unsigned Reg,
                                     const llvm::APInt &Value) const override {
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
};

} // namespace

static ExegesisTarget *getTheExegesisX86Target() {
  static ExegesisX86Target Target;
  return &Target;
}

void InitializeX86ExegesisTarget() {
  ExegesisTarget::registerTarget(getTheExegesisX86Target());
}

} // namespace exegesis
} // namespace llvm
