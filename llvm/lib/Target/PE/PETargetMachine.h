#include "llvm/CodeGen/CodeGenTargetMachineImpl.h"
#include "PESubtarget.h"


namespace llvm{

class PETargetMachine : public CodeGenTargetMachineImpl
{
    std::unique_ptr<TargetLoweringObjectFile> TLOF;
    PESubtarget Subtarget;
public:
    PETargetMachine(const Target &T, const Triple &TT,
    StringRef CPU, StringRef FS,
    const TargetOptions &Options,
    std::optional<Reloc::Model> RM,
    std::optional<CodeModel::Model> CM,
    CodeGenOptLevel OL, bool JIT);

    const PESubtarget *getSubtargetImpl() const {return &Subtarget;}
    
    const PESubtarget *getSubtargetImpl(const Function &F) const override{return &Subtarget;}
    TargetLoweringObjectFile *getObjFileLowering() const override
    {
        return TLOF.get();
    }
    TargetPassConfig *createPassConfig(PassManagerBase &PM) override;
};

}