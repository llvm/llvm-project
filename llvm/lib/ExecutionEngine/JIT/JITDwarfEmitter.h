//===------ JITDwarfEmitter.h - Write dwarf tables into memory ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines a JITDwarfEmitter object that is used by the JIT to
// write dwarf tables to memory.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTION_ENGINE_JIT_DWARFEMITTER_H
#define LLVM_EXECUTION_ENGINE_JIT_DWARFEMITTER_H

namespace llvm {

class Function;
class JITCodeEmitter;
class MachineFunction;
class MachineModuleInfo;
class MachineMove;
class MCAsmInfo;
class DataLayout;
class TargetMachine;
class TargetRegisterInfo;

class JITDwarfEmitter {
  const DataLayout* TD;
  JITCodeEmitter* JCE;
  const TargetRegisterInfo* RI;
  const MCAsmInfo *MAI;
  MachineModuleInfo* MMI;
  JIT& Jit;
  bool stackGrowthDirection;

  unsigned char* EmitExceptionTable(MachineFunction* MF,
                                    unsigned char* StartFunction,
                                    unsigned char* EndFunction) const;

  void EmitFrameMoves(intptr_t BaseLabelPtr,
                      const std::vector<MachineMove> &Moves) const;

  unsigned char* EmitCommonEHFrame(const Function* Personality) const;

  unsigned char* EmitEHFrame(const Function* Personality,
                             unsigned char* StartBufferPtr,
                             unsigned char* StartFunction,
                             unsigned char* EndFunction,
                             unsigned char* ExceptionTable) const;

public:

  JITDwarfEmitter(JIT& jit);

  unsigned char* EmitDwarfTable(MachineFunction& F,
                                JITCodeEmitter& JCE,
                                unsigned char* StartFunction,
                                unsigned char* EndFunction,
                                unsigned char* &EHFramePtr);


  void setModuleInfo(MachineModuleInfo* Info) {
    MMI = Info;
  }
};


} // end namespace llvm

#endif // LLVM_EXECUTION_ENGINE_JIT_DWARFEMITTER_H
