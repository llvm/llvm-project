#include <llvm/CodeGen/MachineFunctionPass.h>

#include "AMDGPU.h"

using namespace llvm;

namespace {
#define DEBUG_TYPE "amdgpu-if-cvt"
const char PassName[] = "AMDGPU if conversion";

class AMDGPUIfConverter : public MachineFunctionPass {
public:
  static char ID;

  AMDGPUIfConverter() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override { return false; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  StringRef getPassName() const override { return PassName; }
};

char AMDGPUIfConverter::ID = 0;

} // namespace

char &llvm::AMDGPUIfConverterID = AMDGPUIfConverter::ID;
INITIALIZE_PASS_BEGIN(AMDGPUIfConverter, DEBUG_TYPE, PassName, false, false)
INITIALIZE_PASS_END(AMDGPUIfConverter, DEBUG_TYPE, PassName, false, false)