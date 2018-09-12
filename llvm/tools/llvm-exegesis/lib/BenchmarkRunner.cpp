//===-- BenchmarkRunner.cpp -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <array>
#include <string>

#include "Assembler.h"
#include "BenchmarkRunner.h"
#include "MCInstrDescView.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Program.h"

namespace exegesis {

BenchmarkFailure::BenchmarkFailure(const llvm::Twine &S)
    : llvm::StringError(S, llvm::inconvertibleErrorCode()) {}

BenchmarkRunner::BenchmarkRunner(const LLVMState &State,
                                 InstructionBenchmark::ModeE Mode)
    : State(State), RATC(State.getRegInfo(),
                         getFunctionReservedRegs(State.getTargetMachine())),
      Mode(Mode), Scratch(llvm::make_unique<ScratchSpace>()) {}

BenchmarkRunner::~BenchmarkRunner() = default;

llvm::Expected<std::vector<InstructionBenchmark>>
BenchmarkRunner::run(unsigned Opcode, unsigned NumRepetitions) {
  const llvm::MCInstrDesc &InstrDesc = State.getInstrInfo().get(Opcode);
  // Ignore instructions that we cannot run.
  if (InstrDesc.isPseudo())
    return llvm::make_error<BenchmarkFailure>("Unsupported opcode: isPseudo");
  if (InstrDesc.isBranch() || InstrDesc.isIndirectBranch())
    return llvm::make_error<BenchmarkFailure>(
        "Unsupported opcode: isBranch/isIndirectBranch");
  if (InstrDesc.isCall() || InstrDesc.isReturn())
    return llvm::make_error<BenchmarkFailure>(
        "Unsupported opcode: isCall/isReturn");

  llvm::Expected<std::vector<BenchmarkCode>> ConfigurationOrError =
      generateConfigurations(Opcode);

  if (llvm::Error E = ConfigurationOrError.takeError())
    return std::move(E);

  std::vector<InstructionBenchmark> InstrBenchmarks;
  for (const BenchmarkCode &Conf : ConfigurationOrError.get())
    InstrBenchmarks.push_back(runConfiguration(Conf, NumRepetitions));
  return InstrBenchmarks;
}

// Repeat the snippet until there are at least NumInstructions in the resulting
// code.
static std::vector<llvm::MCInst>
GenerateInstructions(const BenchmarkCode &BC, const int MinInstructions) {
  std::vector<llvm::MCInst> Code = BC.Instructions;
  for (int I = 0; I < MinInstructions; ++I)
    Code.push_back(BC.Instructions[I % BC.Instructions.size()]);
  return Code;
}

InstructionBenchmark
BenchmarkRunner::runConfiguration(const BenchmarkCode &BC,
                                  unsigned NumRepetitions) const {
  InstructionBenchmark InstrBenchmark;
  InstrBenchmark.Mode = Mode;
  InstrBenchmark.CpuName = State.getTargetMachine().getTargetCPU();
  InstrBenchmark.LLVMTriple =
      State.getTargetMachine().getTargetTriple().normalize();
  InstrBenchmark.NumRepetitions = NumRepetitions;
  InstrBenchmark.Info = BC.Info;

  const std::vector<llvm::MCInst> &Instructions = BC.Instructions;
  if (Instructions.empty()) {
    InstrBenchmark.Error = "Empty snippet";
    return InstrBenchmark;
  }

  InstrBenchmark.Key.Instructions = Instructions;

  // Assemble at least kMinInstructionsForSnippet instructions by repeating the
  // snippet for debug/analysis. This is so that the user clearly understands
  // that the inside instructions are repeated.
  constexpr const int kMinInstructionsForSnippet = 16;
  {
    auto ObjectFilePath = writeObjectFile(
        BC, GenerateInstructions(BC, kMinInstructionsForSnippet));
    if (llvm::Error E = ObjectFilePath.takeError()) {
      InstrBenchmark.Error = llvm::toString(std::move(E));
      return InstrBenchmark;
    }
    const ExecutableFunction EF(State.createTargetMachine(),
                                getObjectFromFile(*ObjectFilePath));
    const auto FnBytes = EF.getFunctionBytes();
    InstrBenchmark.AssembledSnippet.assign(FnBytes.begin(), FnBytes.end());
  }

  // Assemble NumRepetitions instructions repetitions of the snippet for
  // measurements.
  auto ObjectFilePath = writeObjectFile(
      BC, GenerateInstructions(BC, InstrBenchmark.NumRepetitions));
  if (llvm::Error E = ObjectFilePath.takeError()) {
    InstrBenchmark.Error = llvm::toString(std::move(E));
    return InstrBenchmark;
  }
  llvm::outs() << "Check generated assembly with: /usr/bin/objdump -d "
               << *ObjectFilePath << "\n";
  const ExecutableFunction EF(State.createTargetMachine(),
                              getObjectFromFile(*ObjectFilePath));
  InstrBenchmark.Measurements = runMeasurements(EF, *Scratch, NumRepetitions);

  return InstrBenchmark;
}

llvm::Expected<std::vector<BenchmarkCode>>
BenchmarkRunner::generateConfigurations(unsigned Opcode) const {
  if (auto E = generateCodeTemplate(Opcode)) {
    CodeTemplate &CT = E.get();
    std::vector<BenchmarkCode> Output;
    // TODO: Generate as many BenchmarkCode as needed.
    {
      BenchmarkCode BC;
      BC.Info = CT.Info;
      for (InstructionBuilder &IB : CT.Instructions) {
        IB.randomizeUnsetVariables(
            CT.ScratchSpacePointerInReg
                ? RATC.getRegister(CT.ScratchSpacePointerInReg).aliasedBits()
                : RATC.emptyRegisters());
        BC.Instructions.push_back(IB.build());
      }
      if (CT.ScratchSpacePointerInReg)
        BC.LiveIns.push_back(CT.ScratchSpacePointerInReg);
      BC.RegsToDef = computeRegsToDef(CT.Instructions);
      Output.push_back(std::move(BC));
    }
    return Output;
  } else
    return E.takeError();
}

std::vector<unsigned> BenchmarkRunner::computeRegsToDef(
    const std::vector<InstructionBuilder> &Instructions) const {
  // Collect all register uses and create an assignment for each of them.
  // Ignore memory operands which are handled separately.
  // Loop invariant: DefinedRegs[i] is true iif it has been set at least once
  // before the current instruction.
  llvm::BitVector DefinedRegs = RATC.emptyRegisters();
  std::vector<unsigned> RegsToDef;
  for (const InstructionBuilder &IB : Instructions) {
    // Returns the register that this Operand sets or uses, or 0 if this is not
    // a register.
    const auto GetOpReg = [&IB](const Operand &Op) -> unsigned {
      if (Op.IsMem)
        return 0;
      if (Op.ImplicitReg)
        return *Op.ImplicitReg;
      if (Op.IsExplicit && IB.getValueFor(Op).isReg())
        return IB.getValueFor(Op).getReg();
      return 0;
    };
    // Collect used registers that have never been def'ed.
    for (const Operand &Op : IB.Instr.Operands) {
      if (!Op.IsDef) {
        const unsigned Reg = GetOpReg(Op);
        if (Reg > 0 && !DefinedRegs.test(Reg)) {
          RegsToDef.push_back(Reg);
          DefinedRegs.set(Reg);
        }
      }
    }
    // Mark defs as having been def'ed.
    for (const Operand &Op : IB.Instr.Operands) {
      if (Op.IsDef) {
        const unsigned Reg = GetOpReg(Op);
        if (Reg > 0)
          DefinedRegs.set(Reg);
      }
    }
  }
  return RegsToDef;
}

llvm::Expected<std::string>
BenchmarkRunner::writeObjectFile(const BenchmarkCode &BC,
                                 llvm::ArrayRef<llvm::MCInst> Code) const {
  int ResultFD = 0;
  llvm::SmallString<256> ResultPath;
  if (llvm::Error E = llvm::errorCodeToError(llvm::sys::fs::createTemporaryFile(
          "snippet", "o", ResultFD, ResultPath)))
    return std::move(E);
  llvm::raw_fd_ostream OFS(ResultFD, true /*ShouldClose*/);
  assembleToStream(State.getExegesisTarget(), State.createTargetMachine(),
                   BC.LiveIns, BC.RegsToDef, Code, OFS);
  return ResultPath.str();
}

llvm::Expected<CodeTemplate> BenchmarkRunner::generateSelfAliasingCodeTemplate(
    const Instruction &Instr) const {
  const AliasingConfigurations SelfAliasing(Instr, Instr);
  if (SelfAliasing.empty()) {
    return llvm::make_error<BenchmarkFailure>("empty self aliasing");
  }
  CodeTemplate CT;
  InstructionBuilder IB(Instr);
  if (SelfAliasing.hasImplicitAliasing()) {
    CT.Info = "implicit Self cycles, picking random values.";
  } else {
    CT.Info = "explicit self cycles, selecting one aliasing Conf.";
    // This is a self aliasing instruction so defs and uses are from the same
    // instance, hence twice IB in the following call.
    setRandomAliasing(SelfAliasing, IB, IB);
  }
  CT.Instructions.push_back(std::move(IB));
  return std::move(CT);
}

llvm::Expected<CodeTemplate>
BenchmarkRunner::generateUnconstrainedCodeTemplate(const Instruction &Instr,
                                                   llvm::StringRef Msg) const {
  CodeTemplate CT;
  CT.Info = llvm::formatv("{0}, repeating an unconstrained assignment", Msg);
  CT.Instructions.emplace_back(Instr);
  return std::move(CT);
}
} // namespace exegesis
