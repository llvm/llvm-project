//===--------------------- BackendPrinter.h ---------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file implements class BackendPrinter.
/// BackendPrinter is able to collect statistics related to the code executed
/// by the Backend class. Information is then printed out with the help of
/// a MCInstPrinter (to pretty print MCInst objects) and other helper classes.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_MCA_BACKENDPRINTER_H
#define LLVM_TOOLS_LLVM_MCA_BACKENDPRINTER_H

#include "Backend.h"
#include "BackendStatistics.h"
#include "ResourcePressureView.h"
#include "TimelineView.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/ToolOutputFile.h"

#define DEBUG_TYPE "llvm-mca"

namespace mca {

class ResourcePressureView;
class TimelineView;

/// \brief A printer class that knows how to collects statistics on the
/// code analyzed by the llvm-mca tool.
///
/// This class knows how to print out the analysis information collected
/// during the execution of the code. Internally, it delegates to other
/// classes the task of printing out timeline information as well as
/// resource pressure.
class BackendPrinter {
  Backend &B;
  bool EnableVerboseOutput;

  std::unique_ptr<llvm::MCInstPrinter> MCIP;
  std::unique_ptr<llvm::ToolOutputFile> File;

  std::unique_ptr<ResourcePressureView> RPV;
  std::unique_ptr<TimelineView> TV;
  std::unique_ptr<BackendStatistics> BS;

  using Histogram = std::map<unsigned, unsigned>;
  void printDUStatistics(const Histogram &Stats, unsigned Cycles) const;
  void printDispatchStalls(unsigned RATStalls, unsigned RCUStalls,
                           unsigned SQStalls, unsigned LDQStalls,
                           unsigned STQStalls, unsigned DGStalls) const;
  void printRATStatistics(unsigned Mappings, unsigned MaxUsedMappings) const;
  void printRCUStatistics(const Histogram &Histogram, unsigned Cycles) const;
  void printIssuePerCycle(const Histogram &IssuePerCycle,
                          unsigned TotalCycles) const;
  void printSchedulerUsage(const llvm::MCSchedModel &SM,
                           const llvm::ArrayRef<BufferUsageEntry> &Usage) const;
  void printGeneralStatistics(unsigned Iterations, unsigned Cycles,
                              unsigned Instructions,
                              unsigned DispatchWidth) const;
  void printInstructionInfo() const;

  std::unique_ptr<llvm::ToolOutputFile> getOutputStream(std::string OutputFile);
  void initialize(std::string OputputFileName);

public:
  BackendPrinter(Backend &backend, std::string OutputFileName,
                 std::unique_ptr<llvm::MCInstPrinter> IP, bool EnableVerbose)
      : B(backend), EnableVerboseOutput(EnableVerbose), MCIP(std::move(IP)) {
    initialize(OutputFileName);
  }

  ~BackendPrinter() {
    if (File)
      File->keep();
  }

  bool isFileValid() const { return File.get(); }
  llvm::raw_ostream &getOStream() const {
    assert(isFileValid());
    return File->os();
  }

  llvm::MCInstPrinter &getMCInstPrinter() const { return *MCIP; }

  void addResourcePressureView();
  void addTimelineView(unsigned MaxIterations = 3, unsigned MaxCycles = 80);

  void printReport() const;
};

} // namespace mca

#endif
