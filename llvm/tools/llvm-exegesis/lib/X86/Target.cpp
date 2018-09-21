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

namespace exegesis {

namespace {

// Common code for X86 Uops and Latency runners.
template <typename Impl> class X86SnippetGenerator : public Impl {
  using Impl::Impl;

  llvm::Expected<CodeTemplate>
  generateCodeTemplate(unsigned Opcode) const override {
    // Test whether we can generate a snippet for this instruction.
    const auto &InstrInfo = this->State.getInstrInfo();
    const auto OpcodeName = InstrInfo.getName(Opcode);
    if (OpcodeName.startswith("POPF") || OpcodeName.startswith("PUSHF") ||
        OpcodeName.startswith("ADJCALLSTACK")) {
      return llvm::make_error<BenchmarkFailure>(
          "Unsupported opcode: Push/Pop/AdjCallStack");
    }

    // Handle X87.
    const auto &InstrDesc = InstrInfo.get(Opcode);
    const unsigned FPInstClass = InstrDesc.TSFlags & llvm::X86II::FPTypeMask;
    const Instruction Instr(InstrDesc, this->RATC);
    switch (FPInstClass) {
    case llvm::X86II::NotFP:
      break;
    case llvm::X86II::ZeroArgFP:
      return llvm::make_error<BenchmarkFailure>("Unsupported x87 ZeroArgFP");
    case llvm::X86II::OneArgFP:
      return llvm::make_error<BenchmarkFailure>("Unsupported x87 OneArgFP");
    case llvm::X86II::OneArgFPRW:
    case llvm::X86II::TwoArgFP: {
      // These are instructions like
      //   - `ST(0) = fsqrt(ST(0))` (OneArgFPRW)
      //   - `ST(0) = ST(0) + ST(i)` (TwoArgFP)
      // They are intrinsically serial and do not modify the state of the stack.
      // We generate the same code for latency and uops.
      return this->generateSelfAliasingCodeTemplate(Instr);
    }
    case llvm::X86II::CompareFP:
      return Impl::handleCompareFP(Instr);
    case llvm::X86II::CondMovFP:
      return Impl::handleCondMovFP(Instr);
    case llvm::X86II::SpecialFP:
      return llvm::make_error<BenchmarkFailure>("Unsupported x87 SpecialFP");
    default:
      llvm_unreachable("Unknown FP Type!");
    }

    // Fallback to generic implementation.
    return Impl::Base::generateCodeTemplate(Opcode);
  }
};

class X86LatencyImpl : public LatencySnippetGenerator {
protected:
  using Base = LatencySnippetGenerator;
  using Base::Base;
  llvm::Expected<CodeTemplate> handleCompareFP(const Instruction &Instr) const {
    return llvm::make_error<SnippetGeneratorFailure>(
        "Unsupported x87 CompareFP");
  }
  llvm::Expected<CodeTemplate> handleCondMovFP(const Instruction &Instr) const {
    return llvm::make_error<SnippetGeneratorFailure>(
        "Unsupported x87 CondMovFP");
  }
};

class X86UopsImpl : public UopsSnippetGenerator {
protected:
  using Base = UopsSnippetGenerator;
  using Base::Base;
  // We can compute uops for any FP instruction that does not grow or shrink the
  // stack (either do not touch the stack or push as much as they pop).
  llvm::Expected<CodeTemplate> handleCompareFP(const Instruction &Instr) const {
    return generateUnconstrainedCodeTemplate(
        Instr, "instruction does not grow/shrink the FP stack");
  }
  llvm::Expected<CodeTemplate> handleCondMovFP(const Instruction &Instr) const {
    return generateUnconstrainedCodeTemplate(
        Instr, "instruction does not grow/shrink the FP stack");
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
  explicit ConstantInliner(const llvm::APInt &Constant)
      : StackSize(Constant.getBitWidth() / 8) {
    assert(Constant.getBitWidth() % 8 == 0 && "Must be a multiple of 8");
    add(allocateStackSpace(StackSize));
    size_t ByteOffset = 0;
    for (; StackSize - ByteOffset >= 4; ByteOffset += 4)
      add(fillStackSpace(
          llvm::X86::MOV32mi, ByteOffset,
          Constant.extractBits(32, ByteOffset * 8).getZExtValue()));
    if (StackSize - ByteOffset >= 2) {
      add(fillStackSpace(
          llvm::X86::MOV16mi, ByteOffset,
          Constant.extractBits(16, ByteOffset * 8).getZExtValue()));
      ByteOffset += 2;
    }
    if (StackSize - ByteOffset >= 1)
      add(fillStackSpace(
          llvm::X86::MOV8mi, ByteOffset,
          Constant.extractBits(8, ByteOffset * 8).getZExtValue()));
  }

  std::vector<llvm::MCInst> loadAndFinalize(unsigned Reg, unsigned RegBitWidth,
                                            unsigned Opcode) {
    assert(StackSize * 8 == RegBitWidth &&
           "Value does not have the correct size");
    add(loadToReg(Reg, Opcode));
    add(releaseStackSpace(StackSize));
    return std::move(Instructions);
  }

  std::vector<llvm::MCInst>
  loadX87AndFinalize(unsigned Reg, unsigned RegBitWidth, unsigned Opcode) {
    assert(StackSize * 8 == RegBitWidth &&
           "Value does not have the correct size");
    add(llvm::MCInstBuilder(Opcode)
            .addReg(llvm::X86::RSP) // BaseReg
            .addImm(1)              // ScaleAmt
            .addReg(0)              // IndexReg
            .addImm(0)              // Disp
            .addReg(0));            // Segment
    if (Reg != llvm::X86::ST0)
      add(llvm::MCInstBuilder(llvm::X86::ST_Frr).addReg(Reg));
    add(releaseStackSpace(StackSize));
    return std::move(Instructions);
  }

  std::vector<llvm::MCInst> popFlagAndFinalize() {
    assert(StackSize * 8 == 64 && "Value does not have the correct size");
    add(llvm::MCInstBuilder(llvm::X86::POPF64));
    return std::move(Instructions);
  }

private:
  ConstantInliner &add(const llvm::MCInst &Inst) {
    Instructions.push_back(Inst);
    return *this;
  }

  const size_t StackSize;
  std::vector<llvm::MCInst> Instructions;
};

class ExegesisX86Target : public ExegesisTarget {
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

  void fillMemoryOperands(InstructionBuilder &IB, unsigned Reg,
                          unsigned Offset) const override {
    // FIXME: For instructions that read AND write to memory, we use the same
    // value for input and output.
    for (size_t I = 0, E = IB.Instr.Operands.size(); I < E; ++I) {
      const Operand *Op = &IB.Instr.Operands[I];
      if (Op->IsExplicit && Op->IsMem) {
        // Case 1: 5-op memory.
        assert((I + 5 <= E) && "x86 memory references are always 5 ops");
        IB.getValueFor(*Op) = llvm::MCOperand::createReg(Reg); // BaseReg
        Op = &IB.Instr.Operands[++I];
        assert(Op->IsMem);
        assert(Op->IsExplicit);
        IB.getValueFor(*Op) = llvm::MCOperand::createImm(1); // ScaleAmt
        Op = &IB.Instr.Operands[++I];
        assert(Op->IsMem);
        assert(Op->IsExplicit);
        IB.getValueFor(*Op) = llvm::MCOperand::createReg(0); // IndexReg
        Op = &IB.Instr.Operands[++I];
        assert(Op->IsMem);
        assert(Op->IsExplicit);
        IB.getValueFor(*Op) = llvm::MCOperand::createImm(Offset); // Disp
        Op = &IB.Instr.Operands[++I];
        assert(Op->IsMem);
        assert(Op->IsExplicit);
        IB.getValueFor(*Op) = llvm::MCOperand::createReg(0); // Segment
        // Case2: segment:index addressing. We assume that ES is 0.
      }
    }
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
      if (Value.getBitWidth() == 32)
        return CI.loadX87AndFinalize(Reg, 32, llvm::X86::LD_F32m);
      if (Value.getBitWidth() == 64)
        return CI.loadX87AndFinalize(Reg, 64, llvm::X86::LD_F64m);
      if (Value.getBitWidth() == 80)
        return CI.loadX87AndFinalize(Reg, 80, llvm::X86::LD_F80m);
    }
    if (Reg == llvm::X86::EFLAGS)
      return CI.popFlagAndFinalize();
    return {}; // Not yet implemented.
  }

  std::unique_ptr<SnippetGenerator>
  createLatencySnippetGenerator(const LLVMState &State) const override {
    return llvm::make_unique<X86SnippetGenerator<X86LatencyImpl>>(State);
  }

  std::unique_ptr<SnippetGenerator>
  createUopsSnippetGenerator(const LLVMState &State) const override {
    return llvm::make_unique<X86SnippetGenerator<X86UopsImpl>>(State);
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
