#include "Genpay.h"
#include "GenpayTargetMachine.h"
#include "MCTargetDesc/GenpayMCInstLower.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/TargetRegistry.h"

using namespace llvm;

#define DEBUG_TYPE "genpay-asm-printer"

namespace {
class GenpayAsmPrinter : public AsmPrinter {
public:
  static char ID;
  GenpayAsmPrinter(TargetMachine &TM, std::unique_ptr<MCStreamer> Streamer)
      : AsmPrinter(TM, std::move(Streamer), ID) {}

  StringRef getPassName() const override { return "Genpay Assembly Printer"; }

  void emitInstruction(const MachineInstr *MI) override;
};
} // namespace

char GenpayAsmPrinter::ID = 0;

void GenpayAsmPrinter::emitInstruction(const MachineInstr *MI) {
  MCInst LoweredMI;
  GenpayMCInstLower MCInstLowering(OutContext, *this);
  MCInstLowering.Lower(MI, LoweredMI);
  EmitToStreamer(*OutStreamer, LoweredMI);
}

INITIALIZE_PASS(GenpayAsmPrinter, "genpay-asm-printer",
                "Genpay Assembly Printer", false, false)

extern "C" LLVM_ABI LLVM_EXTERNAL_VISIBILITY void
LLVMInitializeGenpayAsmPrinter() {
  RegisterAsmPrinter<GenpayAsmPrinter> X(getTheGenpayTarget());
}
