#ifndef LLVM_LIB_TARGET_InArch_InArch_H
#define LLVM_LIB_TARGET_InArch_InArch_H

#include "MCTargetDesc/InArchMCTargetDesc.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Support/raw_ostream.h"

#define INARCH_DUMP(Color)                                                        \
  {                                                                            \
    llvm::errs().changeColor(Color)                                            \
        << __func__ << "\n\t\t" << __FILE__ << ":" << __LINE__ << "\n";        \
    llvm::errs().changeColor(llvm::raw_ostream::WHITE);                        \
  }
// #define INARCH_DUMP(Color) {}

#define INARCH_DUMP_RED INARCH_DUMP(llvm::raw_ostream::RED)
#define INARCH_DUMP_GREEN INARCH_DUMP(llvm::raw_ostream::GREEN)
#define INARCH_DUMP_YELLOW INARCH_DUMP(llvm::raw_ostream::YELLOW)
#define INARCH_DUMP_CYAN INARCH_DUMP(llvm::raw_ostream::CYAN)
#define INARCH_DUMP_MAGENTA INARCH_DUMP(llvm::raw_ostream::MAGENTA)
#define INARCH_DUMP_WHITE INARCH_DUMP(llvm::raw_ostream::WHITE)

namespace llvm {
class InArchTargetMachine;
class FunctionPass;
class InArchSubtarget;
class AsmPrinter;
class InstructionSelector;
class MCInst;
class MCOperand;
class MachineInstr;
class MachineOperand;
class PassRegistry;

bool lowerInArchMachineInstrToMCInst(const MachineInstr *MI, MCInst &OutMI,
                                    AsmPrinter &AP);
bool LowerInArchMachineOperandToMCOperand(const MachineOperand &MO,
                                         MCOperand &MCOp, const AsmPrinter &AP);

FunctionPass *createInArchISelDag(InArchTargetMachine &TM);
namespace InArch {
enum {
  RA = InArch::R0,
  SP = InArch::R1,
  FP = InArch::R2,
  BP = InArch::R3,
  GP = InArch::R4,
};

} // namespace Sim
} // namespace llvm

#endif // LLVM_LIB_TARGET_InArch_InArch_H
