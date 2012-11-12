//===-- MipsJITInfo.cpp - Implement the Mips JIT Interface ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the JIT interfaces for the Mips target.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "jit"
#include "MipsJITInfo.h"
#include "MipsInstrInfo.h"
#include "MipsRelocations.h"
#include "MipsSubtarget.h"
#include "llvm/Function.h"
#include "llvm/CodeGen/JITCodeEmitter.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Memory.h"
#include <cstdlib>
using namespace llvm;


void MipsJITInfo::replaceMachineCodeForFunction(void *Old, void *New) {
  unsigned NewAddr = (intptr_t)New;
  unsigned OldAddr = (intptr_t)Old;
  const unsigned NopInstr = 0x0;

  // If the functions are in the same memory segment, insert PC-region branch.
  if ((NewAddr & 0xF0000000) == ((OldAddr + 4) & 0xF0000000)) {
    unsigned *OldInstruction = (unsigned *)Old;
    *OldInstruction = 0x08000000;
    unsigned JTargetAddr = NewAddr & 0x0FFFFFFC;

    JTargetAddr >>= 2;
    *OldInstruction |= JTargetAddr;

    // Insert a NOP.
    OldInstruction++;
    *OldInstruction = NopInstr;

    sys::Memory::InvalidateInstructionCache(Old, 2 * 4);
  } else {
    // We need to clear hint bits from the instruction, in case it is 'jr ra'.
    const unsigned HintMask = 0xFFFFF83F, ReturnSequence = 0x03e00008;
    unsigned* CurrentInstr = (unsigned*)Old;
    unsigned CurrInstrHintClear = (*CurrentInstr) & HintMask;
    unsigned* NextInstr = CurrentInstr + 1;
    unsigned NextInstrHintClear = (*NextInstr) & HintMask;

    // Do absolute jump if there are 2 or more instructions before return from
    // the old function.
    if ((CurrInstrHintClear != ReturnSequence) &&
        (NextInstrHintClear != ReturnSequence)) {
      const unsigned LuiT0Instr = 0x3c080000, AddiuT0Instr = 0x25080000;
      const unsigned JrT0Instr = 0x01000008;
      // lui  t0,  high 16 bit of the NewAddr
      (*(CurrentInstr++)) = LuiT0Instr | ((NewAddr & 0xffff0000) >> 16);
      // addiu  t0, t0, low 16 bit of the NewAddr
      (*(CurrentInstr++)) = AddiuT0Instr | (NewAddr & 0x0000ffff);
      // jr t0
      (*(CurrentInstr++)) = JrT0Instr;
      (*CurrentInstr) = NopInstr;

      sys::Memory::InvalidateInstructionCache(Old, 4 * 4);
    } else {
      // Unsupported case
      report_fatal_error("MipsJITInfo::replaceMachineCodeForFunction");
    }
  }
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
// it, because the prolog/epilog inserted by GCC won't work for us. Instead,
// write our own wrapper, which does things our way, so we have complete control
// over register saving and restoring. This code saves registers, calls
// MipsCompilationCallbackC and restores registers.
extern "C" {
#if defined (__mips__)
void MipsCompilationCallback();

  asm(
    ".text\n"
    ".align 2\n"
    ".globl " ASMPREFIX "MipsCompilationCallback\n"
    ASMPREFIX "MipsCompilationCallback:\n"
    ".ent " ASMPREFIX "MipsCompilationCallback\n"
    ".frame  $sp, 32, $ra\n"
    ".set  noreorder\n"
    ".cpload $t9\n"

    "addiu $sp, $sp, -64\n"
    ".cprestore 16\n"

    // Save argument registers a0, a1, a2, a3, f12, f14 since they may contain
    // stuff for the real target function right now. We have to act as if this
    // whole compilation callback doesn't exist as far as the caller is
    // concerned. We also need to save the ra register since it contains the
    // original return address, and t8 register since it contains the address
    // of the end of function stub.
    "sw $a0, 20($sp)\n"
    "sw $a1, 24($sp)\n"
    "sw $a2, 28($sp)\n"
    "sw $a3, 32($sp)\n"
    "sw $ra, 36($sp)\n"
    "sw $t8, 40($sp)\n"
    "sdc1 $f12, 48($sp)\n"
    "sdc1 $f14, 56($sp)\n"

    // t8 points at the end of function stub. Pass the beginning of the stub
    // to the MipsCompilationCallbackC.
    "addiu $a0, $t8, -16\n"
    "jal " ASMPREFIX "MipsCompilationCallbackC\n"
    "nop\n"

    // Restore registers.
    "lw $a0, 20($sp)\n"
    "lw $a1, 24($sp)\n"
    "lw $a2, 28($sp)\n"
    "lw $a3, 32($sp)\n"
    "lw $ra, 36($sp)\n"
    "lw $t8, 40($sp)\n"
    "ldc1 $f12, 48($sp)\n"
    "ldc1 $f14, 56($sp)\n"
    "addiu $sp, $sp, 64\n"

    // Jump to the (newly modified) stub to invoke the real function.
    "addiu $t8, $t8, -16\n"
    "jr $t8\n"
    "nop\n"

    ".set  reorder\n"
    ".end " ASMPREFIX "MipsCompilationCallback\n"
      );
#else  // host != Mips
  void MipsCompilationCallback() {
    llvm_unreachable(
      "Cannot call MipsCompilationCallback() on a non-Mips arch!");
  }
#endif
}

/// MipsCompilationCallbackC - This is the target-specific function invoked
/// by the function stub when we did not know the real target of a call.
/// This function must locate the start of the stub or call site and pass
/// it into the JIT compiler function.
extern "C" void MipsCompilationCallbackC(intptr_t StubAddr) {
  // Get the address of the compiled code for this function.
  intptr_t NewVal = (intptr_t) JITCompilerFunction((void*) StubAddr);

  // Rewrite the function stub so that we don't end up here every time we
  // execute the call. We're replacing the first four instructions of the
  // stub with code that jumps to the compiled function:
  //   lui $t9, %hi(NewVal)
  //   addiu $t9, $t9, %lo(NewVal)
  //   jr $t9
  //   nop

  int Hi = ((unsigned)NewVal & 0xffff0000) >> 16;
  if ((NewVal & 0x8000) != 0)
    Hi++;
  int Lo = (int)(NewVal & 0xffff);

  *(intptr_t *)(StubAddr) = 0xf << 26 | 25 << 16 | Hi;
  *(intptr_t *)(StubAddr + 4) = 9 << 26 | 25 << 21 | 25 << 16 | Lo;
  *(intptr_t *)(StubAddr + 8) = 25 << 21 | 8;
  *(intptr_t *)(StubAddr + 12) = 0;

  sys::Memory::InvalidateInstructionCache((void*) StubAddr, 16);
}

TargetJITInfo::LazyResolverFn MipsJITInfo::getLazyResolverFunction(
    JITCompilerFn F) {
  JITCompilerFunction = F;
  return MipsCompilationCallback;
}

TargetJITInfo::StubLayout MipsJITInfo::getStubLayout() {
  // The stub contains 4 4-byte instructions, aligned at 4 bytes. See
  // emitFunctionStub for details.
  StubLayout Result = { 4*4, 4 };
  return Result;
}

void *MipsJITInfo::emitFunctionStub(const Function *F, void *Fn,
                                    JITCodeEmitter &JCE) {
  JCE.emitAlignment(4);
  void *Addr = (void*) (JCE.getCurrentPCValue());
  if (!sys::Memory::setRangeWritable(Addr, 16))
    llvm_unreachable("ERROR: Unable to mark stub writable.");

  intptr_t EmittedAddr;
  if (Fn != (void*)(intptr_t)MipsCompilationCallback)
    EmittedAddr = (intptr_t)Fn;
  else
    EmittedAddr = (intptr_t)MipsCompilationCallback;


  int Hi = ((unsigned)EmittedAddr & 0xffff0000) >> 16;
  if ((EmittedAddr & 0x8000) != 0)
    Hi++;
  int Lo = (int)(EmittedAddr & 0xffff);

  // lui t9, %hi(EmittedAddr)
  // addiu t9, t9, %lo(EmittedAddr)
  // jalr t8, t9
  // nop
  JCE.emitWordLE(0xf << 26 | 25 << 16 | Hi);
  JCE.emitWordLE(9 << 26 | 25 << 21 | 25 << 16 | Lo);
  JCE.emitWordLE(25 << 21 | 24 << 11 | 9);
  JCE.emitWordLE(0);

  sys::Memory::InvalidateInstructionCache(Addr, 16);
  if (!sys::Memory::setRangeExecutable(Addr, 16))
    llvm_unreachable("ERROR: Unable to mark stub executable.");

  return Addr;
}

/// relocate - Before the JIT can run a block of code that has been emitted,
/// it must rewrite the code to contain the actual addresses of any
/// referenced global symbols.
void MipsJITInfo::relocate(void *Function, MachineRelocation *MR,
                           unsigned NumRelocs, unsigned char *GOTBase) {
  for (unsigned i = 0; i != NumRelocs; ++i, ++MR) {

    void *RelocPos = (char*) Function + MR->getMachineCodeOffset();
    intptr_t ResultPtr = (intptr_t) MR->getResultPointer();

    switch ((Mips::RelocationType) MR->getRelocationType()) {
    case Mips::reloc_mips_pc16:
      ResultPtr = (((ResultPtr - (intptr_t) RelocPos) - 4) >> 2) & 0xffff;
      *((unsigned*) RelocPos) |= (unsigned) ResultPtr;
      break;

    case Mips::reloc_mips_26:
      ResultPtr = (ResultPtr & 0x0fffffff) >> 2;
      *((unsigned*) RelocPos) |= (unsigned) ResultPtr;
      break;

    case Mips::reloc_mips_hi:
      ResultPtr = ResultPtr >> 16;
      if ((((intptr_t) (MR->getResultPointer()) & 0xffff) >> 15) == 1) {
        ResultPtr += 1;
      }
      *((unsigned*) RelocPos) |= (unsigned) ResultPtr;
      break;

    case Mips::reloc_mips_lo: {
      // Addend is needed for unaligned load/store instructions, where offset
      // for the second load/store in the expanded instruction sequence must
      // be modified by +1 or +3. Otherwise, Addend is 0.
      int Addend = *((unsigned*) RelocPos) & 0xffff;
      ResultPtr = (ResultPtr + Addend) & 0xffff;
      *((unsigned*) RelocPos) &= 0xffff0000;
      *((unsigned*) RelocPos) |= (unsigned) ResultPtr;
      break;
    }
    }
  }
}
