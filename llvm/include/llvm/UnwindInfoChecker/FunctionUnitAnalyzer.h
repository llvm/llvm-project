#ifndef LLVM_MC_CFI_ANALYSIS_FUNCTION_UNIT_ANALYZER_H
#define LLVM_MC_CFI_ANALYSIS_FUNCTION_UNIT_ANALYZER_H

#include "llvm/ADT/ArrayRef.h"
#include <cstdio>

namespace llvm {

class MCCFIInstruction;
class MCContext;
class MCInst;

class FunctionUnitAnalyzer {
private:
  MCContext &Context;

public:
  FunctionUnitAnalyzer(const FunctionUnitAnalyzer &) = delete;
  FunctionUnitAnalyzer &operator=(const FunctionUnitAnalyzer &) = delete;
  virtual ~FunctionUnitAnalyzer();

  FunctionUnitAnalyzer(MCContext &Context) : Context(Context) {}

  MCContext &getContext() const { return Context; }

  virtual void startFunctionUnit(bool IsEH,
                                 ArrayRef<MCCFIInstruction> Prologue);
  virtual void
  emitInstructionAndDirectives(const MCInst &Inst,
                               ArrayRef<MCCFIInstruction> Directives);
  virtual void finishFunctionUnit();
};

} // namespace llvm

#endif