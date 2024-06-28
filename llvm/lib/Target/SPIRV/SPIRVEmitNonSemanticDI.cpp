#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Metadata.h"
#include "llvm/PassRegistry.h"
#include "llvm/Support/Casting.h"

namespace llvm {
struct SPIRVEmitNonSemanticDI : public MachineFunctionPass {
  static char ID;
  SPIRVEmitNonSemanticDI();

  bool runOnMachineFunction(MachineFunction &MF) override;
};

void initializeSPIRVEmitNonSemanticDIPass(PassRegistry &);

FunctionPass *createSPIRVEmitNonSemanticDIPass() {
  return new SPIRVEmitNonSemanticDI();
}
} // namespace llvm

using namespace llvm;

INITIALIZE_PASS(SPIRVEmitNonSemanticDI, "spirv-nonsemantic-debug-info",
                "SPIRV NonSemantic.Shader.DebugInfo.100 emitter", false, false)

char SPIRVEmitNonSemanticDI::ID = 0;

SPIRVEmitNonSemanticDI::SPIRVEmitNonSemanticDI() : MachineFunctionPass(ID) {
  initializeSPIRVEmitNonSemanticDIPass(*PassRegistry::getPassRegistry());
}

[[maybe_unused]]
static void findCompileUnitDI(const MachineFunction &MF) {
  MachineModuleInfo &MMI = MF.getMMI();
  const Module *M = MMI.getModule();
  NamedMDNode *DbgCu = M->getNamedMetadata("llvm.dbg.cu");
  std::string FilePath;
  if (DbgCu) {
    unsigned NumOp = DbgCu->getNumOperands();
    if (NumOp) {
      if (const auto *CompileUnit =
              dyn_cast<DICompileUnit>(DbgCu->getOperand(0))) {
        DIFile *File = CompileUnit->getFile();
        FilePath = ((File->getDirectory() + "/" + File->getFilename())).str();
      }
    }
  }
}

bool SPIRVEmitNonSemanticDI::runOnMachineFunction(MachineFunction &MF) {
  return false;
}
