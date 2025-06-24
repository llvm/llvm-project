#include "llvm/UnwindInfoChecker/FunctionUnitAnalyzer.h"

using namespace llvm;

FunctionUnitAnalyzer::~FunctionUnitAnalyzer() = default;

void FunctionUnitAnalyzer::startFunctionUnit(
    bool IsEH, ArrayRef<MCCFIInstruction> Prologue) {}

void FunctionUnitAnalyzer::emitInstructionAndDirectives(
    const MCInst &Inst, ArrayRef<MCCFIInstruction> Directives) {}

void FunctionUnitAnalyzer::finishFunctionUnit() {}