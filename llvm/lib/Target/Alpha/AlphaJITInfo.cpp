//===-- AlphaJITInfo.cpp - Implement the JIT interfaces for the Alpha ---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the JIT interfaces for the Alpha target.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "jit"
#include "AlphaJITInfo.h"
#include "AlphaRelocations.h"
#include "llvm/Function.h"
#include "llvm/CodeGen/MachineCodeEmitter.h"
#include "llvm/Config/alloca.h"
#include "llvm/Support/Debug.h"
#include <cstdlib>
#include <map>
using namespace llvm;

#define BUILD_OFormatI(Op, RA, LIT, FUN, RC) \
  ((Op << 26) | (RA << 21) | (LIT << 13) | (1 << 12) | (FUN << 5) | (RC))
#define BUILD_OFormat(Op, RA, RB, FUN, RC) \
  ((Op << 26) | (RA << 21) | (RB << 16) | (FUN << 5) | (RC))

#define BUILD_LDA(RD, RS, IMM16) \
  ((0x08 << 26) | ((RD) << 21) | ((RS) << 16) | ((IMM16) & 65535))
#define BUILD_LDAH(RD, RS, IMM16) \
  ((0x09 << 26) | ((RD) << 21) | ((RS) << 16) | ((IMM16) & 65535))

#define BUILD_LDQ(RD, RS, IMM16) \
  ((0x29 << 26) | ((RD) << 21) | ((RS) << 16) | ((IMM16) & 0xFFFF))

#define BUILD_JMP(RD, RS, IMM16) \
  ((0x1A << 26) | ((RD) << 21) | ((RS) << 16) | (0x00 << 14) | ((IMM16) & 0x3FFF))
#define BUILD_JSR(RD, RS, IMM16) \
  ((0x1A << 26) | ((RD) << 21) | ((RS) << 16) | (0x01 << 14) | ((IMM16) & 0x3FFF))

#define BUILD_SLLi(RD, RS, IMM8) \
  (BUILD_OFormatI(0x12, RS, IMM8, 0x39, RD))

#define BUILD_ORi(RD, RS, IMM8) \
  (BUILD_OFormatI(0x11, RS, IMM8, 0x20, RD))

#define BUILD_OR(RD, RS, RT) \
  (BUILD_OFormat(0x11, RS, RT, 0x20, RD))



static void EmitBranchToAt(void *At, void *To) {
  unsigned long Fn = (unsigned long)To;

  unsigned *AtI = (unsigned*)At;

  AtI[0] = BUILD_OR(0, 27, 27);

  DOUT << "Stub targeting " << To << "\n";

  for (int x = 1; x <= 8; ++x) {
    AtI[2*x - 1] = BUILD_SLLi(27,27,8);
    unsigned d = (Fn >> (64 - 8 * x)) & 0x00FF;
    //DOUT << "outputing " << hex << d << dec << "\n";
    AtI[2*x] = BUILD_ORi(27, 27, d);
  }
  AtI[17] = BUILD_JMP(31,27,0); //jump, preserving ra, and setting pv
  AtI[18] = 0x00FFFFFF; //mark this as a stub
}

void AlphaJITInfo::replaceMachineCodeForFunction(void *Old, void *New) {
  //FIXME
  assert(0);
}

static TargetJITInfo::JITCompilerFn JITCompilerFunction;
//static AlphaJITInfo* AlphaJTI;

extern "C" {
#ifdef __alpha

  void AlphaCompilationCallbackC(long* oldpv, void* CameFromStub)
  {
    void* Target = JITCompilerFunction(CameFromStub);

    //rewrite the stub to an unconditional branch
    if (((unsigned*)CameFromStub)[18] == 0x00FFFFFF) {
      DOUT << "Came from a stub, rewriting\n";
      EmitBranchToAt(CameFromStub, Target);
    } else {
      DOUT << "confused, didn't come from stub at " << CameFromStub
           << " old jump vector " << oldpv
           << " new jump vector " << Target << "\n";
    }

    //Change pv to new Target
    *oldpv = (long)Target;
  }

  void AlphaCompilationCallback(void);

  asm(
      ".text\n"
      ".globl AlphaComilationCallbackC\n"
      ".align 4\n"
      ".globl AlphaCompilationCallback\n"
      ".ent AlphaCompilationCallback\n"
"AlphaCompilationCallback:\n"
      //      //get JIT's GOT
      "ldgp $29, 0($27)\n"
      //Save args, callee saved, and perhaps others?
      //args: $16-$21 $f16-$f21     (12)
      //callee: $9-$14 $f2-$f9      (14)
      //others: fp:$15 ra:$26 pv:$27 (3)
      "lda $30, -232($30)\n"
      "stq $16,   0($30)\n"
      "stq $17,   8($30)\n"
      "stq $18,  16($30)\n"
      "stq $19,  24($30)\n"
      "stq $20,  32($30)\n"
      "stq $21,  40($30)\n"
      "stt $f16, 48($30)\n"
      "stt $f17, 56($30)\n"
      "stt $f18, 64($30)\n"
      "stt $f19, 72($30)\n"
      "stt $f20, 80($30)\n"
      "stt $f21, 88($30)\n"
      "stq $9,   96($30)\n"
      "stq $10, 104($30)\n"
      "stq $11, 112($30)\n"
      "stq $12, 120($30)\n"
      "stq $13, 128($30)\n"
      "stq $14, 136($30)\n"
      "stt $f2, 144($30)\n"
      "stt $f3, 152($30)\n"
      "stt $f4, 160($30)\n"
      "stt $f5, 168($30)\n"
      "stt $f6, 176($30)\n"
      "stt $f7, 184($30)\n"
      "stt $f8, 192($30)\n"
      "stt $f9, 200($30)\n"
      "stq $15, 208($30)\n"
      "stq $26, 216($30)\n"
      "stq $27, 224($30)\n"

      "addq $30, 224, $16\n" //pass the addr of saved pv as the first arg
      "bis $0, $0, $17\n" //pass the roughly stub addr in second arg
      "jsr $26, AlphaCompilationCallbackC\n" //call without saving ra

      "ldq $16,   0($30)\n"
      "ldq $17,   8($30)\n"
      "ldq $18,  16($30)\n"
      "ldq $19,  24($30)\n"
      "ldq $20,  32($30)\n"
      "ldq $21,  40($30)\n"
      "ldt $f16, 48($30)\n"
      "ldt $f17, 56($30)\n"
      "ldt $f18, 64($30)\n"
      "ldt $f19, 72($30)\n"
      "ldt $f20, 80($30)\n"
      "ldt $f21, 88($30)\n"
      "ldq $9,   96($30)\n"
      "ldq $10, 104($30)\n"
      "ldq $11, 112($30)\n"
      "ldq $12, 120($30)\n"
      "ldq $13, 128($30)\n"
      "ldq $14, 136($30)\n"
      "ldt $f2, 144($30)\n"
      "ldt $f3, 152($30)\n"
      "ldt $f4, 160($30)\n"
      "ldt $f5, 168($30)\n"
      "ldt $f6, 176($30)\n"
      "ldt $f7, 184($30)\n"
      "ldt $f8, 192($30)\n"
      "ldt $f9, 200($30)\n"
      "ldq $15, 208($30)\n"
      "ldq $26, 216($30)\n"
      "ldq $27, 224($30)\n" //this was updated in the callback with the target

      "lda $30, 232($30)\n" //restore sp
      "jmp $31, ($27)\n" //jump to the new function
      ".end AlphaCompilationCallback\n"
      );
#else
  void AlphaCompilationCallback() {
    cerr << "Cannot call AlphaCompilationCallback() on a non-Alpha arch!\n";
    abort();
  }
#endif
}

void *AlphaJITInfo::emitFunctionStub(const Function* F, void *Fn,
                                     MachineCodeEmitter &MCE) {
  //assert(Fn == AlphaCompilationCallback && "Where are you going?\n");
  //Do things in a stupid slow way!
  MCE.startGVStub(F, 19*4);
  void* Addr = (void*)(intptr_t)MCE.getCurrentPCValue();
  for (int x = 0; x < 19; ++ x)
    MCE.emitWordLE(0);
  EmitBranchToAt(Addr, Fn);
  DOUT << "Emitting Stub to " << Fn << " at [" << Addr << "]\n";
  return MCE.finishGVStub(F);
}

TargetJITInfo::LazyResolverFn
AlphaJITInfo::getLazyResolverFunction(JITCompilerFn F) {
  JITCompilerFunction = F;
  //  setZerothGOTEntry((void*)AlphaCompilationCallback);
  return AlphaCompilationCallback;
}

//These describe LDAx
static const int IMM_LOW  = -32768;
static const int IMM_HIGH = 32767;
static const int IMM_MULT = 65536;

static long getUpper16(long l)
{
  long y = l / IMM_MULT;
  if (l % IMM_MULT > IMM_HIGH)
    ++y;
  if (l % IMM_MULT < IMM_LOW)
    --y;
  assert((short)y == y && "displacement out of range");
  return y;
}

static long getLower16(long l)
{
  long h = getUpper16(l);
  long y = l - h * IMM_MULT;
  assert(y == (short)y && "Displacement out of range");
  return y;
}

void AlphaJITInfo::relocate(void *Function, MachineRelocation *MR,
                            unsigned NumRelocs, unsigned char* GOTBase) {
  //because gpdist are paired and relative to the pc of the first inst,
  //we need to have some state

  static std::map<std::pair<void*, int>, void*> gpdistmap;

  for (unsigned i = 0; i != NumRelocs; ++i, ++MR) {
    unsigned *RelocPos = (unsigned*)Function + MR->getMachineCodeOffset()/4;
    long idx = 0;
    bool doCommon = true;
    switch ((Alpha::RelocationType)MR->getRelocationType()) {
    default: assert(0 && "Unknown relocation type!");
    case Alpha::reloc_literal:
      //This is a LDQl
      idx = MR->getGOTIndex();
      DOUT << "Literal relocation to slot " << idx;
      idx = (idx - GOToffset) * 8;
      DOUT << " offset " << idx << "\n";
      break;
    case Alpha::reloc_gprellow:
      idx = (unsigned char*)MR->getResultPointer() - &GOTBase[GOToffset * 8];
      idx = getLower16(idx);
      DOUT << "gprellow relocation offset " << idx << "\n";
      DOUT << " Pointer is " << (void*)MR->getResultPointer()
           << " GOT is " << (void*)&GOTBase[GOToffset * 8] << "\n";
      break;
    case Alpha::reloc_gprelhigh:
      idx = (unsigned char*)MR->getResultPointer() - &GOTBase[GOToffset * 8];
      idx = getUpper16(idx);
      DOUT << "gprelhigh relocation offset " << idx << "\n";
      DOUT << " Pointer is " << (void*)MR->getResultPointer()
           << " GOT is " << (void*)&GOTBase[GOToffset * 8] << "\n";
      break;
    case Alpha::reloc_gpdist:
      switch (*RelocPos >> 26) {
      case 0x09: //LDAH
        idx = &GOTBase[GOToffset * 8] - (unsigned char*)RelocPos;
        idx = getUpper16(idx);
        DOUT << "LDAH: " << idx << "\n";
        //add the relocation to the map
        gpdistmap[std::make_pair(Function, MR->getConstantVal())] = RelocPos;
        break;
      case 0x08: //LDA
        assert(gpdistmap[std::make_pair(Function, MR->getConstantVal())] &&
               "LDAg without seeing LDAHg");
        idx = &GOTBase[GOToffset * 8] -
          (unsigned char*)gpdistmap[std::make_pair(Function, MR->getConstantVal())];
        idx = getLower16(idx);
        DOUT << "LDA: " << idx << "\n";
        break;
      default:
        assert(0 && "Cannot handle gpdist yet");
      }
      break;
    case Alpha::reloc_bsr: {
      idx = (((unsigned char*)MR->getResultPointer() -
             (unsigned char*)RelocPos) >> 2) + 1; //skip first 2 inst of fun
      *RelocPos |= (idx & ((1 << 21)-1));
      doCommon = false;
      break;
    }
    }
    if (doCommon) {
      short x = (short)idx;
      assert(x == idx);
      *(short*)RelocPos = x;
    }
  }
}
