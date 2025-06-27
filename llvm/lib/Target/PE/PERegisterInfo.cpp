/* --- PERegisterInfo.cpp --- */

/* ------------------------------------------
author: 高宇翔
date: 4/1/2025
------------------------------------------ */

#include "PERegisterInfo.h"
#include "MCTargetDesc/PEMCTargetDesc.h"
#include "PE.h"
#include "PESubtarget.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
using namespace llvm;

#define GET_REGINFO_TARGET_DESC
#include "PEGenRegisterInfo.inc"

PERegisterInfo::PERegisterInfo(const PESubtarget &STI) : PEGenRegisterInfo(PE::RS4), STI(STI) {}

PERegisterInfo::~PERegisterInfo() {
  // Destructor
}

const MCPhysReg * PERegisterInfo::getCalleeSavedRegs(const MachineFunction *MF) const {return CSR_SaveList;}

// BitVector代表一个位向量0&1
BitVector PERegisterInfo::getReservedRegs(const MachineFunction &MF) const {
  BitVector Reserved(getNumRegs());
  Reserved.set(PE::RS1);
  Reserved.set(PE::RS0);
  Reserved.set(PE::SP);
  return Reserved;
}

bool PERegisterInfo::eliminateFrameIndex(MachineBasicBlock::iterator II,int SPAdj, unsigned FIOperandNum,RegScavenger *RS) const {

  // 获取当前指令
  MachineInstr &MI = *II;
  // 获取当前指令的操作数(找到带有栈帧的操作数)
  uint i = 0;
  while (!MI.getOperand(i).isFI()) {
    i++;
    assert(i < MI.getNumOperands() && "Instr doesn't have FrameIndex operand!");
  }
  // llvm::errs() << "MI: ";
  // MI.dump();
  // llvm::errs() << "NumOperands: " << MI.getNumOperands() << "\n";

  const int FI = MI.getOperand(i).getIndex(); // 获取栈帧索引

  const MachineFunction &MF = *MI.getParent()->getParent(); // 获取当前函数
  const MachineFrameInfo &MFI = MF.getFrameInfo();          // 获取帧信息

  int64_t Offset = MFI.getObjectOffset(FI); // 获取栈帧偏移量
  uint64_t StackSize = ROUND_UP(
      MFI.getStackSize(),
      STI.getFrameLowering()
          ->getStackAlignment()); // 获取栈大小,保持16位对齐（暂时为16位）

  Offset += static_cast<int64_t>(StackSize); // 得到最终index

  MI.getOperand(i).ChangeToRegister(PE::SP, false); // 将操作数改为栈指针
  // 如果下一个操作数存在且是立即数，则替换为偏移量
  if (i + 1 < MI.getNumOperands() && MI.getOperand(i + 1).isImm()) {
    int64_t O = MI.getOperand(i + 1).getImm();
    Offset += O; // 将偏移量加上操作数的偏移量,这是针对数组中的偏移量
    MI.getOperand(i + 1).ChangeToImmediate(Offset);
  }
  return true;
}

Register PERegisterInfo::getFrameRegister(const MachineFunction &MF) const {
  return PE::SP;
}