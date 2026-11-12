#include "X86TestBase.h"
#include "MCTargetDesc/X86MCTargetDesc.h"
#include "llvm/MC/MCInstBuilder.h"
#include "llvm/Support/TargetSelect.h"

using namespace llvm;
using namespace mca;

X86TestBase::X86TestBase() : MCATestBase("x86_64-unknown-linux", "skylake") {
  LLVMInitializeX86TargetInfo();
  LLVMInitializeX86TargetMC();
  LLVMInitializeX86Target();
  LLVMInitializeX86AsmPrinter();
}

void X86TestBase::getSimpleInsts(SmallVectorImpl<MCInst> &Insts,
                                 unsigned Repeats) {
  for (unsigned i = 0U; i < Repeats; ++i) {
    // vmulps  %xmm0, %xmm1, %xmm2
    Insts.push_back(MCInstBuilder(X86::VMULPSrr)
                        .addReg(X86::XMM2)
                        .addReg(X86::XMM1)
                        .addReg(X86::XMM0));
    // vhaddps %xmm2, %xmm2, %xmm3
    Insts.push_back(MCInstBuilder(X86::VHADDPSrr)
                        .addReg(X86::XMM3)
                        .addReg(X86::XMM2)
                        .addReg(X86::XMM2));
    // vhaddps %xmm3, %xmm3, %xmm4
    Insts.push_back(MCInstBuilder(X86::VHADDPSrr)
                        .addReg(X86::XMM4)
                        .addReg(X86::XMM3)
                        .addReg(X86::XMM3));
  }
}
