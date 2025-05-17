#include "llvm/MC/TargetRegistry.h"
#include "TargetInfo/PETargetInfo.h"
#include "llvm/CodeGen/TargetLoweringObjectFileImpl.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "PETargetMachine.h"
#include "PE.h"

using namespace llvm;

extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializePETarget() {
    RegisterTargetMachine<PETargetMachine>X(getPETarget());

    auto *PR = PassRegistry::getPassRegistry();
    initializePEDAGToDAGISelLegacyPass(*PR);
}

static StringRef computeDataLayout(const Triple&TT,const TargetOptions &Options){
/*
target datalayout = "e-m:e-p:32:32-i64:64-n32-S128"

e       ⼩端序,  E 表示大端序
m:e     表示elf格式,  m:o表示Mach-O格式
p:32:32 表示指针是size是32bit，align是32bit
i64:64  i64类型，使用64bit对齐
n32     ⽬标CPU的原⽣整型是32⽐特
S128    栈以128⽐特⾃然对⻬

target triple = "one-apple-macosx14.0.0"
one: ⽬标架构为one架构
apple: 供应商为Apple
macosx14.0.0: ⽬标操作系统为macOS 14.0


详细参考：https://llvm.org/docs/LangRef.html#langref-datalayout
*/
    assert(TT.isArch32Bit() && "only 32bit");//因为只有32位的后端

    return "e-m:e-p:32:32-i64:64-n32-S128";
}

static Reloc::Model getEffectiveRelocModel(const Triple &TT,
    std::optional<Reloc::Model> RM) {
return RM.value_or(Reloc::Static);
}

PETargetMachine::PETargetMachine(const Target &T, const Triple &TT,
    StringRef CPU, StringRef FS,
    const TargetOptions &Options,
    std::optional<Reloc::Model> RM,
    std::optional<CodeModel::Model> CM,
    CodeGenOptLevel OL, bool JIT)
: CodeGenTargetMachineImpl(T, computeDataLayout(TT, Options), TT, CPU, FS,
Options, getEffectiveRelocModel(TT, RM),
getEffectiveCodeModel(CM, CodeModel::Small), OL),
TLOF(std::make_unique<TargetLoweringObjectFileELF>()),Subtarget(TT, CPU, FS, *this){
    /*
    CodeModel
    small : 支持一个数据段一个代码段，所有的数据和代码紧挨着
	large : 支持多个代码段，多个数据段
	tiny  : 只支持运行在MS-DOS，tiny模型将所有的数据和代码放入一个段中，因此整个程序大小不能超过64k
	medium : 介于small和large之间，支持多个代码段和单个数据段.
    参考链接：http://www.c-jump.com/CIS77/ASM/Directives/D77_0030_models.htm
    */
    initAsmInfo();
}

namespace {
class PEPassConfig : public TargetPassConfig {
public:
    PEPassConfig(PETargetMachine &TM, PassManagerBase &PM)
        : TargetPassConfig(TM, PM) {}
    
    PETargetMachine &getPETargetMachine() const {
        return getTM<PETargetMachine>();
    }
    const PESubtarget &getPESubtarget() const {
        return *getPETargetMachine().getSubtargetImpl();
    }
    bool addInstSelector() override;
};
}

TargetPassConfig *PETargetMachine::createPassConfig(PassManagerBase &PM) {
    return new PEPassConfig(*this, PM);
}

bool PEPassConfig::addInstSelector() {
    addPass(createPEISelDag(getPETargetMachine(), CodeGenOptLevel::None));
    return false;
}