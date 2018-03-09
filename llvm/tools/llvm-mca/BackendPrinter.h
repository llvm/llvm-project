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
///
/// BackendPrinter allows the customization of the performance report.  With the
/// help of this class, users can specify their own custom sequence of views.
/// Each view is then printed out in sequence when method printReport() is
/// called.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_MCA_BACKENDPRINTER_H
#define LLVM_TOOLS_LLVM_MCA_BACKENDPRINTER_H

#include "Backend.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "llvm-mca"

namespace mca {

class View;

/// \brief A printer class that knows how to collects statistics on the
/// code analyzed by the llvm-mca tool.
///
/// This class knows how to print out the analysis information collected
/// during the execution of the code. Internally, it delegates to other
/// classes the task of printing out timeline information as well as
/// resource pressure.
class BackendPrinter {
  const Backend &B;
  llvm::MCInstPrinter &MCIP;
  llvm::SmallVector<std::unique_ptr<View>, 8> Views;

  void printGeneralStatistics(llvm::raw_ostream &OS,
                              unsigned Iterations, unsigned Cycles,
                              unsigned Instructions,
                              unsigned DispatchWidth) const;
  void printInstructionInfo(llvm::raw_ostream &OS) const;

public:
  BackendPrinter(const Backend &backend, llvm::MCInstPrinter &IP)
      : B(backend), MCIP(IP) {}

  llvm::MCInstPrinter &getMCInstPrinter() const { return MCIP; }

  void addView(std::unique_ptr<View> V) { Views.emplace_back(std::move(V)); }
  void printReport(llvm::raw_ostream &OS) const;
};

} // namespace mca

#endif
