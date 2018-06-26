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
#include "MCTargetDesc/X86MCTargetDesc.h"
#include "X86.h"
#include "X86RegisterInfo.h"
#include "llvm/MC/MCInstBuilder.h"

namespace exegesis {

// Test whether we can generate a snippet for this instruction.
static llvm::Error shouldRun(const LLVMState &State, const unsigned Opcode) {
  const auto &InstrInfo = State.getInstrInfo();
  const auto OpcodeName = InstrInfo.getName(Opcode);
  if (OpcodeName.startswith("POPF") || OpcodeName.startswith("PUSHF") ||
      OpcodeName.startswith("ADJCALLSTACK")) {
    return llvm::make_error<BenchmarkFailure>(
        "Unsupported opcode: Push/Pop/AdjCallStack");
  }
  return llvm::ErrorSuccess();
}

namespace {

class X86LatencyBenchmarkRunner : public LatencyBenchmarkRunner {
private:
  using LatencyBenchmarkRunner::LatencyBenchmarkRunner;

  llvm::Expected<SnippetPrototype>
  generatePrototype(unsigned Opcode) const override {
    if (llvm::Error E = shouldRun(State, Opcode)) {
      return std::move(E);
    }
    return LatencyBenchmarkRunner::generatePrototype(Opcode);
  }
};

class X86UopsBenchmarkRunner : public UopsBenchmarkRunner {
private:
  using UopsBenchmarkRunner::UopsBenchmarkRunner;

  llvm::Expected<SnippetPrototype>
  generatePrototype(unsigned Opcode) const override {
    if (llvm::Error E = shouldRun(State, Opcode)) {
      return std::move(E);
    }
    return UopsBenchmarkRunner::generatePrototype(Opcode);
  }
};

class ExegesisX86Target : public ExegesisTarget {
  void addTargetSpecificPasses(llvm::PassManagerBase &PM) const override {
    // Lowers FP pseudo-instructions, e.g. ABS_Fp32 -> ABS_F.
    // FIXME: Enable when the exegesis assembler no longer does
    // Properties.reset(TracksLiveness);
    // PM.add(llvm::createX86FloatingPointStackifierPass());
  }

  std::vector<llvm::MCInst>
  setRegToConstant(const unsigned Reg) const override {
    // FIXME: Handle FP stack:
    // llvm::X86::RFP32RegClass
    // llvm::X86::RFP64RegClass
    // llvm::X86::RFP80RegClass
    if (llvm::X86::GR8RegClass.contains(Reg)) {
      return {llvm::MCInstBuilder(llvm::X86::MOV8ri).addReg(Reg).addImm(1)};
    }
    if (llvm::X86::GR16RegClass.contains(Reg)) {
      return {llvm::MCInstBuilder(llvm::X86::MOV16ri).addReg(Reg).addImm(1)};
    }
    if (llvm::X86::GR32RegClass.contains(Reg)) {
      return {llvm::MCInstBuilder(llvm::X86::MOV32ri).addReg(Reg).addImm(1)};
    }
    if (llvm::X86::GR64RegClass.contains(Reg)) {
      return {llvm::MCInstBuilder(llvm::X86::MOV64ri32).addReg(Reg).addImm(1)};
    }
    if (llvm::X86::VR128XRegClass.contains(Reg)) {
      return setVectorRegToConstant(Reg, 16, llvm::X86::VMOVDQUrm);
    }
    if (llvm::X86::VR256XRegClass.contains(Reg)) {
      return setVectorRegToConstant(Reg, 32, llvm::X86::VMOVDQUYrm);
    }
    if (llvm::X86::VR512RegClass.contains(Reg)) {
      return setVectorRegToConstant(Reg, 64, llvm::X86::VMOVDQU64Zrm);
    }
    return {};
  }

  std::unique_ptr<BenchmarkRunner>
  createLatencyBenchmarkRunner(const LLVMState &State) const override {
    return llvm::make_unique<X86LatencyBenchmarkRunner>(State);
  }

  std::unique_ptr<BenchmarkRunner>
  createUopsBenchmarkRunner(const LLVMState &State) const override {
    return llvm::make_unique<X86UopsBenchmarkRunner>(State);
  }

  bool matchesArch(llvm::Triple::ArchType Arch) const override {
    return Arch == llvm::Triple::x86_64 || Arch == llvm::Triple::x86;
  }

private:
  // setRegToConstant() specialized for a vector register of size
  // `RegSizeBytes`. `RMOpcode` is the opcode used to do a memory -> vector
  // register load.
  static std::vector<llvm::MCInst>
  setVectorRegToConstant(const unsigned Reg, const unsigned RegSizeBytes,
                         const unsigned RMOpcode) {
    // There is no instruction to directly set XMM, go through memory.
    // Since vector values can be interpreted as integers of various sizes (8
    // to 64 bits) as well as floats and double, so we chose an immediate
    // value that has set bits for all byte values and is a normal float/
    // double. 0x40404040 is ~32.5 when interpreted as a double and ~3.0f when
    // interpreted as a float.
    constexpr const uint64_t kImmValue = 0x40404040ull;
    std::vector<llvm::MCInst> Result;
    // Allocate scratch memory on the stack.
    Result.push_back(llvm::MCInstBuilder(llvm::X86::SUB64ri8)
                         .addReg(llvm::X86::RSP)
                         .addReg(llvm::X86::RSP)
                         .addImm(RegSizeBytes));
    // Fill scratch memory.
    for (unsigned Disp = 0; Disp < RegSizeBytes; Disp += 4) {
      Result.push_back(llvm::MCInstBuilder(llvm::X86::MOV32mi)
                           // Address = ESP
                           .addReg(llvm::X86::RSP) // BaseReg
                           .addImm(1)              // ScaleAmt
                           .addReg(0)              // IndexReg
                           .addImm(Disp)           // Disp
                           .addReg(0)              // Segment
                           // Immediate.
                           .addImm(kImmValue));
    }
    // Load Reg from scratch memory.
    Result.push_back(llvm::MCInstBuilder(RMOpcode)
                         .addReg(Reg)
                         // Address = ESP
                         .addReg(llvm::X86::RSP) // BaseReg
                         .addImm(1)              // ScaleAmt
                         .addReg(0)              // IndexReg
                         .addImm(0)              // Disp
                         .addReg(0));            // Segment
    // Release scratch memory.
    Result.push_back(llvm::MCInstBuilder(llvm::X86::ADD64ri8)
                         .addReg(llvm::X86::RSP)
                         .addReg(llvm::X86::RSP)
                         .addImm(RegSizeBytes));
    return Result;
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
