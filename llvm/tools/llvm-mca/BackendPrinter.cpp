//===--------------------- BackendPrinter.cpp -------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file implements the BackendPrinter interface.
///
//===----------------------------------------------------------------------===//

#include "BackendPrinter.h"
#include "llvm/CodeGen/TargetSchedule.h"

namespace mca {

using namespace llvm;

std::unique_ptr<ToolOutputFile>
BackendPrinter::getOutputStream(std::string OutputFile) {
  if (OutputFile == "")
    OutputFile = "-";
  std::error_code EC;
  auto Out = llvm::make_unique<ToolOutputFile>(OutputFile, EC, sys::fs::F_None);
  if (!EC)
    return Out;
  errs() << EC.message() << '\n';
  return nullptr;
}

void BackendPrinter::printGeneralStatistics(unsigned Iterations,
                                            unsigned Cycles,
                                            unsigned Instructions,
                                            unsigned DispatchWidth) const {
  unsigned TotalInstructions = Instructions * Iterations;
  double IPC = (double)TotalInstructions / Cycles;

  std::string Buffer;
  raw_string_ostream TempStream(Buffer);
  TempStream << "Iterations:     " << Iterations;
  TempStream << "\nInstructions:   " << TotalInstructions;
  TempStream << "\nTotal Cycles:   " << Cycles;
  TempStream << "\nDispatch Width: " << DispatchWidth;
  TempStream << "\nIPC:            " << format("%.2f", IPC) << '\n';
  TempStream.flush();
  File->os() << Buffer;
}

void BackendPrinter::printRATStatistics(unsigned TotalMappings,
                                        unsigned MaxUsedMappings) const {
  std::string Buffer;
  raw_string_ostream TempStream(Buffer);
  TempStream << "\n\nRegister Alias Table:";
  TempStream << "\nTotal number of mappings created: " << TotalMappings;
  TempStream << "\nMax number of mappings used:      " << MaxUsedMappings
             << '\n';
  TempStream.flush();
  File->os() << Buffer;
}

void BackendPrinter::printDispatchStalls(unsigned RATStalls, unsigned RCUStalls,
                                         unsigned SCHEDQStalls,
                                         unsigned LDQStalls, unsigned STQStalls,
                                         unsigned DGStalls) const {
  std::string Buffer;
  raw_string_ostream TempStream(Buffer);
  TempStream << "\n\nDynamic Dispatch Stall Cycles:\n";
  TempStream << "RAT     - Register unavailable:                      "
             << RATStalls;
  TempStream << "\nRCU     - Retire tokens unavailable:                 "
             << RCUStalls;
  TempStream << "\nSCHEDQ  - Scheduler full:                            "
             << SCHEDQStalls;
  TempStream << "\nLQ      - Load queue full:                           "
             << LDQStalls;
  TempStream << "\nSQ      - Store queue full:                          "
             << STQStalls;
  TempStream << "\nGROUP   - Static restrictions on the dispatch group: "
             << DGStalls;
  TempStream << '\n';
  TempStream.flush();
  File->os() << Buffer;
}

void BackendPrinter::printSchedulerUsage(
    const MCSchedModel &SM, const ArrayRef<BufferUsageEntry> &Usage) const {
  std::string Buffer;
  raw_string_ostream TempStream(Buffer);
  TempStream << "\n\nScheduler's queue usage:\n";
  const ArrayRef<uint64_t> ResourceMasks = B.getProcResourceMasks();
  for (unsigned I = 0, E = SM.getNumProcResourceKinds(); I < E; ++I) {
    const MCProcResourceDesc &ProcResource = *SM.getProcResource(I);
    if (!ProcResource.BufferSize)
      continue;

    for (const BufferUsageEntry &Entry : Usage)
      if (ResourceMasks[I] == Entry.first)
        TempStream << ProcResource.Name << ",  " << Entry.second << '/'
                   << ProcResource.BufferSize << '\n';
  }

  TempStream.flush();
  File->os() << Buffer;
}

void BackendPrinter::printInstructionInfo() const {
  std::string Buffer;
  raw_string_ostream TempStream(Buffer);

  TempStream << "\n\nInstruction Info:\n";
  TempStream << "[1]: #uOps\n[2]: Latency\n[3]: RThroughput\n"
             << "[4]: MayLoad\n[5]: MayStore\n[6]: HasSideEffects\n\n";

  TempStream << "[1]    [2]    [3]    [4]    [5]    [6]\tInstructions:\n";
  for (unsigned I = 0, E = B.getNumInstructions(); I < E; ++I) {
    const MCInst &Inst = B.getMCInstFromIndex(I);
    const InstrDesc &ID = B.getInstrDesc(Inst);
    unsigned NumMicroOpcodes = ID.NumMicroOps;
    unsigned Latency = ID.MaxLatency;
    double RThroughput = B.getRThroughput(ID);
    TempStream << ' ' << NumMicroOpcodes << "    ";
    if (NumMicroOpcodes < 10)
      TempStream << "  ";
    else if (NumMicroOpcodes < 100)
      TempStream << ' ';
    TempStream << Latency << "   ";
    if (Latency < 10.0)
      TempStream << "  ";
    else if (Latency < 100.0)
      TempStream << ' ';
    if (RThroughput) {
      TempStream << format("%.2f", RThroughput) << ' ';
      if (RThroughput < 10.0)
        TempStream << "  ";
      else if (RThroughput < 100.0)
        TempStream << ' ';
    } else {
      TempStream << " -     ";
    }
    TempStream << (ID.MayLoad ? " *     " : "       ");
    TempStream << (ID.MayStore ? " *     " : "       ");
    TempStream << (ID.HasSideEffects ? " * " : "   ");
    MCIP->printInst(&Inst, TempStream, "", B.getSTI());
    TempStream << '\n';
  }

  TempStream.flush();
  File->os() << Buffer;
}

void BackendPrinter::printReport() const {
  assert(isFileValid());
  unsigned Cycles = B.getNumCycles();
  printGeneralStatistics(B.getNumIterations(), Cycles, B.getNumInstructions(),
                         B.getDispatchWidth());
  if (EnableVerboseOutput) {
    printDispatchStalls(B.getNumRATStalls(), B.getNumRCUStalls(),
                        B.getNumSQStalls(), B.getNumLDQStalls(),
                        B.getNumSTQStalls(), B.getNumDispatchGroupStalls());
    printRATStatistics(B.getTotalRegisterMappingsCreated(),
                       B.getMaxUsedRegisterMappings());
    BS->printHistograms(File->os());

    std::vector<BufferUsageEntry> Usage;
    B.getBuffersUsage(Usage);
    printSchedulerUsage(B.getSchedModel(), Usage);
  }

  if (RPV) {
    RPV->printResourcePressure(getOStream(), Cycles);
    printInstructionInfo();
  }

  if (TV) {
    TV->printTimeline(getOStream());
    TV->printAverageWaitTimes(getOStream());
  }
}

void BackendPrinter::addResourcePressureView() {
  if (!RPV) {
    RPV = llvm::make_unique<ResourcePressureView>(
        B.getSTI(), *MCIP, B.getSourceMgr(), B.getProcResourceMasks());
    B.addEventListener(RPV.get());
  }
}

void BackendPrinter::addTimelineView(unsigned MaxIterations,
                                     unsigned MaxCycles) {
  if (!TV) {
    TV = llvm::make_unique<TimelineView>(B.getSTI(), *MCIP, B.getSourceMgr(),
                                         MaxIterations, MaxCycles);
    B.addEventListener(TV.get());
  }
}

void BackendPrinter::initialize(std::string OutputFileName) {
  File = getOutputStream(OutputFileName);
  MCIP->setPrintImmHex(false);
  if (EnableVerboseOutput) {
    BS = llvm::make_unique<BackendStatistics>();
    B.addEventListener(BS.get());
  }
}

} // namespace mca.
