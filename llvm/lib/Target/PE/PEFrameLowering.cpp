/* --- PEFrameLowering.cpp --- */

/* ------------------------------------------
author: undefined
date: 4/2/2025
------------------------------------------ */

#include "PEFrameLowering.h"
#include "MCTargetDesc/PEMCTargetDesc.h"
#include "PE.h"
#include "PEInstrInfo.h"
#include "PESubtarget.h"
#include "llvm/CodeGen/MachineFrameInfo.h"

using namespace llvm;

// 函数的汇编代码开头序言
void PEFrameLowering::emitPrologue(MachineFunction &MF,
                                   MachineBasicBlock &MBB) const {

  // MachineBasicBlock::iterator MBBI = MBB.begin();
  // const TargetInstrInfo &TII = *STI.getInstrInfo();
  // DebugLoc DL = MBBI != MBB.end() ? MBBI->getDebugLoc() : DebugLoc();
  // int STACKSIZE = computeStackSize(MF); // 计算栈大小

  // if (STACKSIZE == 0)
  //   return;
  // BuildMI(MBB, MBBI, DL, TII.get(PE::ADDI), PE::RS0)
  //     .addReg(PE::RS0)
  //     .addImm(-STACKSIZE)
  //     .setMIFlag(MachineInstr::FrameSetup);
}

// 函数的汇编代码结尾序言
void PEFrameLowering::emitEpilogue(MachineFunction &MF,
                                   MachineBasicBlock &MBB) const {
  // MachineBasicBlock::iterator MBBI = MBB.getLastNonDebugInstr();
  // const TargetInstrInfo &TII = *STI.getInstrInfo();
  // DebugLoc DL = MBBI != MBB.end() ? MBBI->getDebugLoc() : DebugLoc();
  // int STACKSIZE = computeStackSize(MF); // 计算栈大小

  // if (STACKSIZE == 0)
  //   return;
  // BuildMI(MBB, MBBI, DL, TII.get(PE::ADDI), PE::RS0)
  //     .addReg(PE::RS0)
  //     .addImm(STACKSIZE)
  //     .setMIFlag(MachineInstr::FrameDestroy);
}

bool PEFrameLowering::hasFPImpl(const MachineFunction &MF) const {
  return false;
}

uint64_t llvm::PEFrameLowering::computeStackSize(MachineFunction &MF) const {
  uint64_t StackSize =
      MF.getFrameInfo()
          .getStackSize(); // 获取当前函数需要的栈空间大小（单位：字节）这个大小是所有局部变量、溢出寄存器等在栈上的总和，但未必已经对齐
  if (getStackAlignment() > 0) // 返回目标架构要求的栈对齐字节数（如 8、16
                               // 等）。
    StackSize = ROUND_UP(StackSize, getStackAlignment());
  return StackSize;
}

