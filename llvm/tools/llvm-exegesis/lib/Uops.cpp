//===-- Uops.cpp ------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Uops.h"
#include "BenchmarkResult.h"
#include "InstructionSnippetGenerator.h"
#include "PerfHelper.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/MC/MCSchedule.h"
#include "llvm/Support/Error.h"
#include <algorithm>
#include <random>
#include <unordered_map>
#include <unordered_set>

namespace exegesis {

// FIXME: Handle memory (see PR36906)
static bool isInvalidOperand(const llvm::MCOperandInfo &OpInfo) {
  switch (OpInfo.OperandType) {
  default:
    return true;
  case llvm::MCOI::OPERAND_IMMEDIATE:
  case llvm::MCOI::OPERAND_REGISTER:
    return false;
  }
}

static llvm::Error makeError(llvm::Twine Msg) {
  return llvm::make_error<llvm::StringError>(Msg,
                                             llvm::inconvertibleErrorCode());
}

// FIXME: Read the counter names from the ProcResourceUnits when PR36984 is
// fixed.
static const std::string *getEventNameFromProcResName(const char *ProcResName) {
  static const std::unordered_map<std::string, std::string> Entries = {
      {"SBPort0", "UOPS_DISPATCHED_PORT:PORT_0"},
      {"SBPort1", "UOPS_DISPATCHED_PORT:PORT_1"},
      {"SBPort4", "UOPS_DISPATCHED_PORT:PORT_4"},
      {"SBPort5", "UOPS_DISPATCHED_PORT:PORT_5"},
      {"HWPort0", "UOPS_DISPATCHED_PORT:PORT_0"},
      {"HWPort1", "UOPS_DISPATCHED_PORT:PORT_1"},
      {"HWPort2", "UOPS_DISPATCHED_PORT:PORT_2"},
      {"HWPort3", "UOPS_DISPATCHED_PORT:PORT_3"},
      {"HWPort4", "UOPS_DISPATCHED_PORT:PORT_4"},
      {"HWPort5", "UOPS_DISPATCHED_PORT:PORT_5"},
      {"HWPort6", "UOPS_DISPATCHED_PORT:PORT_6"},
      {"HWPort7", "UOPS_DISPATCHED_PORT:PORT_7"},
      {"SKLPort0", "UOPS_DISPATCHED_PORT:PORT_0"},
      {"SKLPort1", "UOPS_DISPATCHED_PORT:PORT_1"},
      {"SKLPort2", "UOPS_DISPATCHED_PORT:PORT_2"},
      {"SKLPort3", "UOPS_DISPATCHED_PORT:PORT_3"},
      {"SKLPort4", "UOPS_DISPATCHED_PORT:PORT_4"},
      {"SKLPort5", "UOPS_DISPATCHED_PORT:PORT_5"},
      {"SKLPort6", "UOPS_DISPATCHED_PORT:PORT_6"},
      {"SKXPort7", "UOPS_DISPATCHED_PORT:PORT_7"},
      {"SKXPort0", "UOPS_DISPATCHED_PORT:PORT_0"},
      {"SKXPort1", "UOPS_DISPATCHED_PORT:PORT_1"},
      {"SKXPort2", "UOPS_DISPATCHED_PORT:PORT_2"},
      {"SKXPort3", "UOPS_DISPATCHED_PORT:PORT_3"},
      {"SKXPort4", "UOPS_DISPATCHED_PORT:PORT_4"},
      {"SKXPort5", "UOPS_DISPATCHED_PORT:PORT_5"},
      {"SKXPort6", "UOPS_DISPATCHED_PORT:PORT_6"},
      {"SKXPort7", "UOPS_DISPATCHED_PORT:PORT_7"},
  };
  const auto It = Entries.find(ProcResName);
  return It == Entries.end() ? nullptr : &It->second;
}

static std::vector<llvm::MCInst> generateIndependentAssignments(
    const LLVMState &State, const llvm::MCInstrDesc &InstrDesc,
    llvm::SmallVector<Variable, 8> Vars, int MaxAssignments) {
  std::unordered_set<llvm::MCPhysReg> IsUsedByAnyVar;
  for (const Variable &Var : Vars) {
    if (Var.IsUse) {
      IsUsedByAnyVar.insert(Var.PossibleRegisters.begin(),
                            Var.PossibleRegisters.end());
    }
  }

  std::vector<llvm::MCInst> Pattern;
  for (int A = 0; A < MaxAssignments; ++A) {
    // FIXME: This is a bit pessimistic. We should get away with an
    // assignment that ensures that the set of assigned registers for uses and
    // the set of assigned registers for defs do not intersect (registers
    // for uses (resp defs) do not have to be all distinct).
    const std::vector<llvm::MCPhysReg> Regs = getExclusiveAssignment(Vars);
    if (Regs.empty())
      break;
    // Remove all assigned registers defs that are used by at least one other
    // variable from the list of possible variable registers. This ensures that
    // we never create a RAW hazard that would lead to serialization.
    for (size_t I = 0, E = Vars.size(); I < E; ++I) {
      llvm::MCPhysReg Reg = Regs[I];
      if (Vars[I].IsDef && IsUsedByAnyVar.count(Reg)) {
        Vars[I].PossibleRegisters.remove(Reg);
      }
    }
    // Create an MCInst and check assembly.
    llvm::MCInst Inst = generateMCInst(InstrDesc, Vars, Regs);
    if (!State.canAssemble(Inst))
      continue;
    Pattern.push_back(std::move(Inst));
  }
  return Pattern;
}

UopsBenchmarkRunner::~UopsBenchmarkRunner() = default;

const char *UopsBenchmarkRunner::getDisplayName() const { return "uops"; }

llvm::Expected<std::vector<llvm::MCInst>> UopsBenchmarkRunner::createCode(
    const LLVMState &State, const unsigned OpcodeIndex,
    const unsigned NumRepetitions, const JitFunctionContext &Context) const {
  const auto &InstrInfo = State.getInstrInfo();
  const auto &RegInfo = State.getRegInfo();
  const llvm::MCInstrDesc &InstrDesc = InstrInfo.get(OpcodeIndex);
  for (const llvm::MCOperandInfo &OpInfo : InstrDesc.operands()) {
    if (isInvalidOperand(OpInfo))
      return makeError("Only registers and immediates are supported");
  }

  // FIXME: Load constants into registers (e.g. with fld1) to not break
  // instructions like x87.

  // Ideally we would like the only limitation on executing uops to be the issue
  // ports. Maximizing port pressure increases the likelihood that the load is
  // distributed evenly across possible ports.

  // To achieve that, one approach is to generate instructions that do not have
  // data dependencies between them.
  //
  // For some instructions, this is trivial:
  //    mov rax, qword ptr [rsi]
  //    mov rax, qword ptr [rsi]
  //    mov rax, qword ptr [rsi]
  //    mov rax, qword ptr [rsi]
  // For the above snippet, haswell just renames rax four times and executes the
  // four instructions two at a time on P23 and P0126.
  //
  // For some instructions, we just need to make sure that the source is
  // different from the destination. For example, IDIV8r reads from GPR and
  // writes to AX. We just need to ensure that the variable is assigned a
  // register which is different from AX:
  //    idiv bx
  //    idiv bx
  //    idiv bx
  //    idiv bx
  // The above snippet will be able to fully saturate the ports, while the same
  // with ax would issue one uop every `latency(IDIV8r)` cycles.
  //
  // Some instructions make this harder because they both read and write from
  // the same register:
  //    inc rax
  //    inc rax
  //    inc rax
  //    inc rax
  // This has a data dependency from each instruction to the next, limit the
  // number of instructions that can be issued in parallel.
  // It turns out that this is not a big issue on recent Intel CPUs because they
  // have heuristics to balance port pressure. In the snippet above, subsequent
  // instructions will end up evenly distributed on {P0,P1,P5,P6}, but some CPUs
  // might end up executing them all on P0 (just because they can), or try
  // avoiding P5 because it's usually under high pressure from vector
  // instructions.
  // This issue is even more important for high-latency instructions because
  // they increase the idle time of the CPU, e.g. :
  //    imul rax, rbx
  //    imul rax, rbx
  //    imul rax, rbx
  //    imul rax, rbx
  //
  // To avoid that, we do the renaming statically by generating as many
  // independent exclusive assignments as possible (until all possible registers
  // are exhausted) e.g.:
  //    imul rax, rbx
  //    imul rcx, rbx
  //    imul rdx, rbx
  //    imul r8,  rbx
  //
  // Some instruction even make the above static renaming impossible because
  // they implicitly read and write from the same operand, e.g. ADC16rr reads
  // and writes from EFLAGS.
  // In that case we just use a greedy register assignment and hope for the
  // best.

  const auto Vars = getVariables(RegInfo, InstrDesc, Context.getReservedRegs());

  // Generate as many independent exclusive assignments as possible.
  constexpr const int MaxStaticRenames = 20;
  std::vector<llvm::MCInst> Pattern =
      generateIndependentAssignments(State, InstrDesc, Vars, MaxStaticRenames);
  if (Pattern.empty()) {
    // We don't even have a single exclusive assignment, fallback to a greedy
    // assignment.
    // FIXME: Tell the user about this decision to help debugging.
    const std::vector<llvm::MCPhysReg> Regs = getGreedyAssignment(Vars);
    if (!Vars.empty() && Regs.empty())
      return makeError("No feasible greedy assignment");
    llvm::MCInst Inst = generateMCInst(InstrDesc, Vars, Regs);
    if (!State.canAssemble(Inst))
      return makeError("Cannot assemble greedy assignment");
    Pattern.push_back(std::move(Inst));
  }

  // Generate repetitions of the pattern until benchmark_iterations is reached.
  std::vector<llvm::MCInst> Result;
  Result.reserve(NumRepetitions);
  for (unsigned I = 0; I < NumRepetitions; ++I)
    Result.push_back(Pattern[I % Pattern.size()]);
  return Result;
}

std::vector<BenchmarkMeasure>
UopsBenchmarkRunner::runMeasurements(const LLVMState &State,
                                     const JitFunction &Function,
                                     const unsigned NumRepetitions) const {
  const auto &SchedModel = State.getSubtargetInfo().getSchedModel();

  std::vector<BenchmarkMeasure> Result;
  for (unsigned ProcResIdx = 1;
       ProcResIdx < SchedModel.getNumProcResourceKinds(); ++ProcResIdx) {
    const llvm::MCProcResourceDesc &ProcRes =
        *SchedModel.getProcResource(ProcResIdx);
    const std::string *const EventName =
        getEventNameFromProcResName(ProcRes.Name);
    if (!EventName)
      continue;
    pfm::Counter Counter{pfm::PerfEvent(*EventName)};
    Counter.start();
    Function();
    Counter.stop();
    Result.push_back({llvm::itostr(ProcResIdx),
                      static_cast<double>(Counter.read()) / NumRepetitions,
                      ProcRes.Name});
  }
  return Result;
}

} // namespace exegesis
