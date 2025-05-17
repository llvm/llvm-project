#include "PEMCTargetDesc.h"
#include "PEMCAsmInfo.h"
#include "PEInstrInfo.h"
#include "PERegisterInfo.h"
#include "PESubtarget.h"
#include "PEInstPrinter.h"
#include "llvm/MC/TargetRegistry.h"
#include "TargetInfo/PETargetInfo.h"

#define GET_INSTRINFO_MC_DESC
#define ENABLE_INSTR_PREDICATE_VERIFIER
#include "PEGenInstrInfo.inc"

#define GET_REGINFO_MC_DESC
#include "PEGenRegisterInfo.inc"

#define GET_SUBTARGETINFO_MC_DESC
#include "PEGenSubtargetInfo.inc"

using namespace llvm;

//注册MC(Machine Code)层核心组件，将PE后端的多个机器码生成模块注册到LLVM全局目标注册表中，包括：
// ​汇编信息 (RegisterMCAsmInfo)：定义汇编语法规则（如指令分隔符、注释符号等）。
// ​目标文件信息 (RegisterMCObjectFileInfo)：指定 ELF 等目标文件格式的生成规则。
// ​指令与寄存器信息 (RegisterMCInstrInfo, RegisterMCRegInfo)：描述 RISC-V 指令集架构（ISA）和寄存器布局。
// ​代码发射器与汇编后端 (RegisterMCCodeEmitter, RegisterMCAsmBackend)：负责将 LLVM IR 转换为机器码并生成汇编或二进制文件。
// ​指令流生成器 (RegisterELFStreamer, RegisterObjectTargetStreamer)：控制目标文件流（如 ELF 文件）的生成逻辑。
static MCAsmInfo *createPEMCAsmInfo(const MCRegisterInfo &MRI,
    const Triple &TT,
    const MCTargetOptions &Options){
        return new PEMCAsmInfo(TT);

    }
static MCRegisterInfo *createPEMCRegisterInfo(const Triple &TT) {
    MCRegisterInfo* X = new MCRegisterInfo();
    InitPEMCRegisterInfo(X,PE::X1);
    return X;
}

static MCInstrInfo *createPEMCInstrInfo() {
    MCInstrInfo* X = new MCInstrInfo();
    InitPEMCInstrInfo(X);
    return X;
}
static MCSubtargetInfo *createPEMCSubtargetInfo(const Triple &TT,
    StringRef CPU, StringRef FS){
    if(CPU.empty())
        CPU = "PE";
    return createPEMCSubtargetInfoImpl(TT,CPU,CPU,FS);
}
static MCInstPrinter *createPEMCInstPrinter(const Triple &T,
    unsigned SyntaxVariant,
    const MCAsmInfo &MAI,
    const MCInstrInfo &MII,
    const MCRegisterInfo &MRI) {
    return new PEInstPrinter(MAI, MII, MRI);
}

extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializePETargetMC() {
    TargetRegistry::RegisterMCAsmInfo(getPETarget(), createPEMCAsmInfo);
    TargetRegistry::RegisterMCRegInfo(getPETarget(), createPEMCRegisterInfo);
    TargetRegistry::RegisterMCInstrInfo(getPETarget(), createPEMCInstrInfo);
    TargetRegistry::RegisterMCSubtargetInfo(getPETarget(), createPEMCSubtargetInfo);//子目标信息注册
    TargetRegistry::RegisterMCInstPrinter(getPETarget(), createPEMCInstPrinter);
} 