//===-- ARMJITInfo.cpp - Implement the JIT interfaces for the ARM target --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the JIT interfaces for the ARM target.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "jit"
#include "ARMJITInfo.h"
#include "ARMInstrInfo.h"
#include "ARMConstantPoolValue.h"
#include "ARMRelocations.h"
#include "ARMSubtarget.h"
#include "llvm/Function.h"
#include "llvm/CodeGen/MachineCodeEmitter.h"
#include "llvm/Config/alloca.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Streams.h"
#include "llvm/System/Memory.h"
#include <cstdlib>
using namespace llvm;

void ARMJITInfo::replaceMachineCodeForFunction(void *Old, void *New) {
  abort();
}

/// JITCompilerFunction - This contains the address of the JIT function used to
/// compile a function lazily.
static TargetJITInfo::JITCompilerFn JITCompilerFunction;

// Get the ASMPREFIX for the current host.  This is often '_'.
#ifndef __USER_LABEL_PREFIX__
#define __USER_LABEL_PREFIX__
#endif
#define GETASMPREFIX2(X) #X
#define GETASMPREFIX(X) GETASMPREFIX2(X)
#define ASMPREFIX GETASMPREFIX(__USER_LABEL_PREFIX__)

// CompilationCallback stub - We can't use a C function with inline assembly in
// it, because we the prolog/epilog inserted by GCC won't work for us (we need
// to preserve more context and manipulate the stack directly).  Instead,
// write our own wrapper, which does things our way, so we have complete 
// control over register saving and restoring.
extern "C" {
#if defined(__arm__)
  void ARMCompilationCallback(void);
  asm(
    ".text\n"
    ".align 2\n"
    ".globl " ASMPREFIX "ARMCompilationCallback\n"
    ASMPREFIX "ARMCompilationCallback:\n"
    // Save caller saved registers since they may contain stuff
    // for the real target function right now. We have to act as if this
    // whole compilation callback doesn't exist as far as the caller is
    // concerned, so we can't just preserve the callee saved regs.
    "stmdb sp!, {r0, r1, r2, r3, lr}\n"
#ifndef __SOFTFP__
    "fstmfdd sp!, {d0, d1, d2, d3, d4, d5, d6, d7}\n"
#endif
    // The LR contains the address of the stub function on entry.
    // pass it as the argument to the C part of the callback
    "mov  r0, lr\n"
    "sub  sp, sp, #4\n"
    // Call the C portion of the callback
    "bl   " ASMPREFIX "ARMCompilationCallbackC\n"
    "add  sp, sp, #4\n"
    // Restoring the LR to the return address of the function that invoked
    // the stub and de-allocating the stack space for it requires us to
    // swap the two saved LR values on the stack, as they're backwards
    // for what we need since the pop instruction has a pre-determined
    // order for the registers.
    //      +--------+
    //   0  | LR     | Original return address
    //      +--------+    
    //   1  | LR     | Stub address (start of stub)
    // 2-5  | R3..R0 | Saved registers (we need to preserve all regs)
    // 6-20 | D0..D7 | Saved VFP registers
    //      +--------+    
    //
#ifndef __SOFTFP__
    // Restore VFP caller-saved registers.
    "fldmfdd sp!, {d0, d1, d2, d3, d4, d5, d6, d7}\n"
#endif
    //
    //      We need to exchange the values in slots 0 and 1 so we can
    //      return to the address in slot 1 with the address in slot 0
    //      restored to the LR.
    "ldr  r0, [sp,#20]\n"
    "ldr  r1, [sp,#16]\n"
    "str  r1, [sp,#20]\n"
    "str  r0, [sp,#16]\n"
    // Return to the (newly modified) stub to invoke the real function.
    // The above twiddling of the saved return addresses allows us to
    // deallocate everything, including the LR the stub saved, all in one
    // pop instruction.
    "ldmia  sp!, {r0, r1, r2, r3, lr, pc}\n"
      );
#else  // Not an ARM host
  void ARMCompilationCallback() {
    assert(0 && "Cannot call ARMCompilationCallback() on a non-ARM arch!\n");
    abort();
  }
#endif
}

/// ARMCompilationCallbackC - This is the target-specific function invoked 
/// by the function stub when we did not know the real target of a call.  
/// This function must locate the start of the stub or call site and pass 
/// it into the JIT compiler function.
extern "C" void ARMCompilationCallbackC(intptr_t StubAddr) {
  // Get the address of the compiled code for this function.
  intptr_t NewVal = (intptr_t)JITCompilerFunction((void*)StubAddr);

  // Rewrite the call target... so that we don't end up here every time we
  // execute the call. We're replacing the first two instructions of the
  // stub with:
  //   ldr pc, [pc,#-4]
  //   <addr>
  if (!sys::Memory::setRangeWritable((void*)StubAddr, 8)) {
    cerr << "ERROR: Unable to mark stub writable\n";
    abort();
  }
  *(intptr_t *)StubAddr = 0xe51ff004;  // ldr pc, [pc, #-4]
  *(intptr_t *)(StubAddr+4) = NewVal;
  if (!sys::Memory::setRangeExecutable((void*)StubAddr, 8)) {
    cerr << "ERROR: Unable to mark stub executable\n";
    abort();
  }
}

TargetJITInfo::LazyResolverFn
ARMJITInfo::getLazyResolverFunction(JITCompilerFn F) {
  JITCompilerFunction = F;
  return ARMCompilationCallback;
}

void *ARMJITInfo::emitGlobalValueIndirectSym(const GlobalValue *GV, void *Ptr,
                                             MachineCodeEmitter &MCE) {
  MCE.startGVStub(GV, 4, 4);
  MCE.emitWordLE((intptr_t)Ptr);
  void *PtrAddr = MCE.finishGVStub(GV);
  addIndirectSymAddr(Ptr, (intptr_t)PtrAddr);
  return PtrAddr;
}

void *ARMJITInfo::emitFunctionStub(const Function* F, void *Fn,
                                   MachineCodeEmitter &MCE) {
  // If this is just a call to an external function, emit a branch instead of a
  // call.  The code is the same except for one bit of the last instruction.
  if (Fn != (void*)(intptr_t)ARMCompilationCallback) {
    // Branch to the corresponding function addr.
    if (IsPIC) {
      // The stub is 8-byte size and 4-aligned.
      intptr_t LazyPtr = getIndirectSymAddr(Fn);
      if (!LazyPtr) {
        // In PIC mode, the function stub is loading a lazy-ptr.
        LazyPtr= (intptr_t)emitGlobalValueIndirectSym((GlobalValue*)F, Fn, MCE);
        if (F)
          DOUT << "JIT: Indirect symbol emitted at [" << LazyPtr << "] for GV '"
               << F->getName() << "'\n";
        else
          DOUT << "JIT: Stub emitted at [" << LazyPtr
               << "] for external function at '" << Fn << "'\n";
      }
      MCE.startGVStub(F, 16, 4);
      intptr_t Addr = (intptr_t)MCE.getCurrentPCValue();
      MCE.emitWordLE(0xe59fc004);            // ldr pc, [pc, #+4]
      MCE.emitWordLE(0xe08fc00c);            // L_func$scv: add ip, pc, ip
      MCE.emitWordLE(0xe59cf000);            // ldr pc, [ip]
      MCE.emitWordLE(LazyPtr - (Addr+4+8));  // func - (L_func$scv+8)
      sys::Memory::InvalidateInstructionCache((void*)Addr, 16);
    } else {
      // The stub is 8-byte size and 4-aligned.
      MCE.startGVStub(F, 8, 4);
      intptr_t Addr = (intptr_t)MCE.getCurrentPCValue();
      MCE.emitWordLE(0xe51ff004);    // ldr pc, [pc, #-4]
      MCE.emitWordLE((intptr_t)Fn);  // addr of function
      sys::Memory::InvalidateInstructionCache((void*)Addr, 8);
    }
  } else {
    // The compilation callback will overwrite the first two words of this
    // stub with indirect branch instructions targeting the compiled code. 
    // This stub sets the return address to restart the stub, so that
    // the new branch will be invoked when we come back.
    //
    // Branch and link to the compilation callback.
    // The stub is 16-byte size and 4-byte aligned.
    MCE.startGVStub(F, 16, 4);
    intptr_t Addr = (intptr_t)MCE.getCurrentPCValue();
    // Save LR so the callback can determine which stub called it.
    // The compilation callback is responsible for popping this prior
    // to returning.
    MCE.emitWordLE(0xe92d4000); // push {lr}
    // Set the return address to go back to the start of this stub.
    MCE.emitWordLE(0xe24fe00c); // sub lr, pc, #12
    // Invoke the compilation callback.
    MCE.emitWordLE(0xe51ff004); // ldr pc, [pc, #-4]
    // The address of the compilation callback.
    MCE.emitWordLE((intptr_t)ARMCompilationCallback);
    sys::Memory::InvalidateInstructionCache((void*)Addr, 16);
  }

  return MCE.finishGVStub(F);
}

intptr_t ARMJITInfo::resolveRelocDestAddr(MachineRelocation *MR) const {
  ARM::RelocationType RT = (ARM::RelocationType)MR->getRelocationType();
  switch (RT) {
  default:
    return (intptr_t)(MR->getResultPointer());
  case ARM::reloc_arm_pic_jt:
    // Destination address - jump table base.
    return (intptr_t)(MR->getResultPointer()) - MR->getConstantVal();
  case ARM::reloc_arm_jt_base:
    // Jump table base address.
    return getJumpTableBaseAddr(MR->getJumpTableIndex());
  case ARM::reloc_arm_cp_entry:
  case ARM::reloc_arm_vfp_cp_entry:
    // Constant pool entry address.
    return getConstantPoolEntryAddr(MR->getConstantPoolIndex());
  case ARM::reloc_arm_machine_cp_entry: {
    ARMConstantPoolValue *ACPV = (ARMConstantPoolValue*)MR->getConstantVal();
    assert((!ACPV->hasModifier() && !ACPV->mustAddCurrentAddress()) &&
           "Can't handle this machine constant pool entry yet!");
    intptr_t Addr = (intptr_t)(MR->getResultPointer());
    Addr -= getPCLabelAddr(ACPV->getLabelId()) + ACPV->getPCAdjustment();
    return Addr;
  }
  }
}

/// relocate - Before the JIT can run a block of code that has been emitted,
/// it must rewrite the code to contain the actual addresses of any
/// referenced global symbols.
void ARMJITInfo::relocate(void *Function, MachineRelocation *MR,
                          unsigned NumRelocs, unsigned char* GOTBase) {
  for (unsigned i = 0; i != NumRelocs; ++i, ++MR) {
    void *RelocPos = (char*)Function + MR->getMachineCodeOffset();
    intptr_t ResultPtr = resolveRelocDestAddr(MR);
    switch ((ARM::RelocationType)MR->getRelocationType()) {
    case ARM::reloc_arm_cp_entry:
    case ARM::reloc_arm_vfp_cp_entry:
    case ARM::reloc_arm_relative: {
      // It is necessary to calculate the correct PC relative value. We
      // subtract the base addr from the target addr to form a byte offset.
      ResultPtr = ResultPtr - (intptr_t)RelocPos - 8;
      // If the result is positive, set bit U(23) to 1.
      if (ResultPtr >= 0)
        *((intptr_t*)RelocPos) |= 1 << ARMII::U_BitShift;
      else {
        // Otherwise, obtain the absolute value and set bit U(23) to 0.
        *((intptr_t*)RelocPos) &= ~(1 << ARMII::U_BitShift);
        ResultPtr = - ResultPtr;
      }
      // Set the immed value calculated.
      // VFP immediate offset is multiplied by 4.
      if (MR->getRelocationType() == ARM::reloc_arm_vfp_cp_entry)
        ResultPtr = ResultPtr >> 2;
      *((intptr_t*)RelocPos) |= ResultPtr;
      // Set register Rn to PC.
      *((intptr_t*)RelocPos) |=
        ARMRegisterInfo::getRegisterNumbering(ARM::PC) << ARMII::RegRnShift;
      break;
    }
    case ARM::reloc_arm_pic_jt:
    case ARM::reloc_arm_machine_cp_entry:
    case ARM::reloc_arm_absolute: {
      // These addresses have already been resolved.
      *((intptr_t*)RelocPos) |= (intptr_t)ResultPtr;
      break;
    }
    case ARM::reloc_arm_branch: {
      // It is necessary to calculate the correct value of signed_immed_24
      // field. We subtract the base addr from the target addr to form a
      // byte offset, which must be inside the range -33554432 and +33554428.
      // Then, we set the signed_immed_24 field of the instruction to bits
      // [25:2] of the byte offset. More details ARM-ARM p. A4-11.
      ResultPtr = ResultPtr - (intptr_t)RelocPos - 8;
      ResultPtr = (ResultPtr & 0x03FFFFFC) >> 2;
      assert(ResultPtr >= -33554432 && ResultPtr <= 33554428);
      *((intptr_t*)RelocPos) |= ResultPtr;
      break;
    }
    case ARM::reloc_arm_jt_base: {
      // JT base - (instruction addr + 8)
      ResultPtr = ResultPtr - (intptr_t)RelocPos - 8;
      *((intptr_t*)RelocPos) |= ResultPtr;
      break;
    }
    }
  }
}
