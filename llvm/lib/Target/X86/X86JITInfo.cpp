//===-- X86JITInfo.cpp - Implement the JIT interfaces for the X86 target --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the JIT interfaces for the X86 target.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "jit"
#include "X86JITInfo.h"
#include "X86Relocations.h"
#include "X86Subtarget.h"
#include "X86TargetMachine.h"
#include "llvm/Function.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Valgrind.h"
#include <cstdlib>
#include <cstring>
using namespace llvm;

// Determine the platform we're running on
#if defined (__x86_64__) || defined (_M_AMD64) || defined (_M_X64)
# define X86_64_JIT
#elif defined(__i386__) || defined(i386) || defined(_M_IX86)
# define X86_32_JIT
#endif

void X86JITInfo::replaceMachineCodeForFunction(void *Old, void *New) {
  unsigned char *OldByte = (unsigned char *)Old;
  *OldByte++ = 0xE9;                // Emit JMP opcode.
  unsigned *OldWord = (unsigned *)OldByte;
  unsigned NewAddr = (intptr_t)New;
  unsigned OldAddr = (intptr_t)OldWord;
  *OldWord = NewAddr - OldAddr - 4; // Emit PC-relative addr of New code.

  // X86 doesn't need to invalidate the processor cache, so just invalidate
  // Valgrind's cache directly.
  sys::ValgrindDiscardTranslations(Old, 5);
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

// For ELF targets, use a .size and .type directive, to let tools
// know the extent of functions defined in assembler.
#if defined(__ELF__)
# define SIZE(sym) ".size " #sym ", . - " #sym "\n"
# define TYPE_FUNCTION(sym) ".type " #sym ", @function\n"
#else
# define SIZE(sym)
# define TYPE_FUNCTION(sym)
#endif

// Provide a convenient way for disabling usage of CFI directives.
// This is needed for old/broken assemblers (for example, gas on
// Darwin is pretty old and doesn't support these directives)
#if defined(__APPLE__)
# define CFI(x)
#else
// FIXME: Disable this until we really want to use it. Also, we will
//        need to add some workarounds for compilers, which support
//        only subset of these directives.
# define CFI(x)
#endif

// Provide a wrapper for X86CompilationCallback2 that saves non-traditional
// callee saved registers, for the fastcc calling convention.
extern "C" {
#if defined(X86_64_JIT)
# ifndef _MSC_VER
  // No need to save EAX/EDX for X86-64.
  void X86CompilationCallback(void);
  asm(
    ".text\n"
    ".align 8\n"
    ".globl " ASMPREFIX "X86CompilationCallback\n"
    TYPE_FUNCTION(X86CompilationCallback)
  ASMPREFIX "X86CompilationCallback:\n"
    CFI(".cfi_startproc\n")
    // Save RBP
    "pushq   %rbp\n"
    CFI(".cfi_def_cfa_offset 16\n")
    CFI(".cfi_offset %rbp, -16\n")
    // Save RSP
    "movq    %rsp, %rbp\n"
    CFI(".cfi_def_cfa_register %rbp\n")
    // Save all int arg registers
    "pushq   %rdi\n"
    CFI(".cfi_rel_offset %rdi, 0\n")
    "pushq   %rsi\n"
    CFI(".cfi_rel_offset %rsi, 8\n")
    "pushq   %rdx\n"
    CFI(".cfi_rel_offset %rdx, 16\n")
    "pushq   %rcx\n"
    CFI(".cfi_rel_offset %rcx, 24\n")
    "pushq   %r8\n"
    CFI(".cfi_rel_offset %r8, 32\n")
    "pushq   %r9\n"
    CFI(".cfi_rel_offset %r9, 40\n")
    // Align stack on 16-byte boundary. ESP might not be properly aligned
    // (8 byte) if this is called from an indirect stub.
    "andq    $-16, %rsp\n"
    // Save all XMM arg registers
    "subq    $128, %rsp\n"
    "movaps  %xmm0, (%rsp)\n"
    "movaps  %xmm1, 16(%rsp)\n"
    "movaps  %xmm2, 32(%rsp)\n"
    "movaps  %xmm3, 48(%rsp)\n"
    "movaps  %xmm4, 64(%rsp)\n"
    "movaps  %xmm5, 80(%rsp)\n"
    "movaps  %xmm6, 96(%rsp)\n"
    "movaps  %xmm7, 112(%rsp)\n"
    // JIT callee
#ifdef _WIN64
    "subq    $32, %rsp\n"
    "movq    %rbp, %rcx\n"    // Pass prev frame and return address
    "movq    8(%rbp), %rdx\n"
    "call    " ASMPREFIX "X86CompilationCallback2\n"
    "addq    $32, %rsp\n"
#else
    "movq    %rbp, %rdi\n"    // Pass prev frame and return address
    "movq    8(%rbp), %rsi\n"
    "call    " ASMPREFIX "X86CompilationCallback2\n"
#endif
    // Restore all XMM arg registers
    "movaps  112(%rsp), %xmm7\n"
    "movaps  96(%rsp), %xmm6\n"
    "movaps  80(%rsp), %xmm5\n"
    "movaps  64(%rsp), %xmm4\n"
    "movaps  48(%rsp), %xmm3\n"
    "movaps  32(%rsp), %xmm2\n"
    "movaps  16(%rsp), %xmm1\n"
    "movaps  (%rsp), %xmm0\n"
    // Restore RSP
    "movq    %rbp, %rsp\n"
    CFI(".cfi_def_cfa_register %rsp\n")
    // Restore all int arg registers
    "subq    $48, %rsp\n"
    CFI(".cfi_adjust_cfa_offset 48\n")
    "popq    %r9\n"
    CFI(".cfi_adjust_cfa_offset -8\n")
    CFI(".cfi_restore %r9\n")
    "popq    %r8\n"
    CFI(".cfi_adjust_cfa_offset -8\n")
    CFI(".cfi_restore %r8\n")
    "popq    %rcx\n"
    CFI(".cfi_adjust_cfa_offset -8\n")
    CFI(".cfi_restore %rcx\n")
    "popq    %rdx\n"
    CFI(".cfi_adjust_cfa_offset -8\n")
    CFI(".cfi_restore %rdx\n")
    "popq    %rsi\n"
    CFI(".cfi_adjust_cfa_offset -8\n")
    CFI(".cfi_restore %rsi\n")
    "popq    %rdi\n"
    CFI(".cfi_adjust_cfa_offset -8\n")
    CFI(".cfi_restore %rdi\n")
    // Restore RBP
    "popq    %rbp\n"
    CFI(".cfi_adjust_cfa_offset -8\n")
    CFI(".cfi_restore %rbp\n")
    "ret\n"
    CFI(".cfi_endproc\n")
    SIZE(X86CompilationCallback)
  );
# else
  // No inline assembler support on this platform. The routine is in external
  // file.
  void X86CompilationCallback();

# endif
#elif defined (X86_32_JIT)
# ifndef _MSC_VER
  void X86CompilationCallback(void);
  asm(
    ".text\n"
    ".align 8\n"
    ".globl " ASMPREFIX "X86CompilationCallback\n"
    TYPE_FUNCTION(X86CompilationCallback)
  ASMPREFIX "X86CompilationCallback:\n"
    CFI(".cfi_startproc\n")
    "pushl   %ebp\n"
    CFI(".cfi_def_cfa_offset 8\n")
    CFI(".cfi_offset %ebp, -8\n")
    "movl    %esp, %ebp\n"    // Standard prologue
    CFI(".cfi_def_cfa_register %ebp\n")
    "pushl   %eax\n"
    CFI(".cfi_rel_offset %eax, 0\n")
    "pushl   %edx\n"          // Save EAX/EDX/ECX
    CFI(".cfi_rel_offset %edx, 4\n")
    "pushl   %ecx\n"
    CFI(".cfi_rel_offset %ecx, 8\n")
#  if defined(__APPLE__)
    "andl    $-16, %esp\n"    // Align ESP on 16-byte boundary
#  endif
    "subl    $16, %esp\n"
    "movl    4(%ebp), %eax\n" // Pass prev frame and return address
    "movl    %eax, 4(%esp)\n"
    "movl    %ebp, (%esp)\n"
    "call    " ASMPREFIX "X86CompilationCallback2\n"
    "movl    %ebp, %esp\n"    // Restore ESP
    CFI(".cfi_def_cfa_register %esp\n")
    "subl    $12, %esp\n"
    CFI(".cfi_adjust_cfa_offset 12\n")
    "popl    %ecx\n"
    CFI(".cfi_adjust_cfa_offset -4\n")
    CFI(".cfi_restore %ecx\n")
    "popl    %edx\n"
    CFI(".cfi_adjust_cfa_offset -4\n")
    CFI(".cfi_restore %edx\n")
    "popl    %eax\n"
    CFI(".cfi_adjust_cfa_offset -4\n")
    CFI(".cfi_restore %eax\n")
    "popl    %ebp\n"
    CFI(".cfi_adjust_cfa_offset -4\n")
    CFI(".cfi_restore %ebp\n")
    "ret\n"
    CFI(".cfi_endproc\n")
    SIZE(X86CompilationCallback)
  );

  // Same as X86CompilationCallback but also saves XMM argument registers.
  void X86CompilationCallback_SSE(void);
  asm(
    ".text\n"
    ".align 8\n"
    ".globl " ASMPREFIX "X86CompilationCallback_SSE\n"
    TYPE_FUNCTION(X86CompilationCallback_SSE)
  ASMPREFIX "X86CompilationCallback_SSE:\n"
    CFI(".cfi_startproc\n")
    "pushl   %ebp\n"
    CFI(".cfi_def_cfa_offset 8\n")
    CFI(".cfi_offset %ebp, -8\n")
    "movl    %esp, %ebp\n"    // Standard prologue
    CFI(".cfi_def_cfa_register %ebp\n")
    "pushl   %eax\n"
    CFI(".cfi_rel_offset %eax, 0\n")
    "pushl   %edx\n"          // Save EAX/EDX/ECX
    CFI(".cfi_rel_offset %edx, 4\n")
    "pushl   %ecx\n"
    CFI(".cfi_rel_offset %ecx, 8\n")
    "andl    $-16, %esp\n"    // Align ESP on 16-byte boundary
    // Save all XMM arg registers
    "subl    $64, %esp\n"
    // FIXME: provide frame move information for xmm registers.
    // This can be tricky, because CFA register is ebp (unaligned)
    // and we need to produce offsets relative to it.
    "movaps  %xmm0, (%esp)\n"
    "movaps  %xmm1, 16(%esp)\n"
    "movaps  %xmm2, 32(%esp)\n"
    "movaps  %xmm3, 48(%esp)\n"
    "subl    $16, %esp\n"
    "movl    4(%ebp), %eax\n" // Pass prev frame and return address
    "movl    %eax, 4(%esp)\n"
    "movl    %ebp, (%esp)\n"
    "call    " ASMPREFIX "X86CompilationCallback2\n"
    "addl    $16, %esp\n"
    "movaps  48(%esp), %xmm3\n"
    CFI(".cfi_restore %xmm3\n")
    "movaps  32(%esp), %xmm2\n"
    CFI(".cfi_restore %xmm2\n")
    "movaps  16(%esp), %xmm1\n"
    CFI(".cfi_restore %xmm1\n")
    "movaps  (%esp), %xmm0\n"
    CFI(".cfi_restore %xmm0\n")
    "movl    %ebp, %esp\n"    // Restore ESP
    CFI(".cfi_def_cfa_register esp\n")
    "subl    $12, %esp\n"
    CFI(".cfi_adjust_cfa_offset 12\n")
    "popl    %ecx\n"
    CFI(".cfi_adjust_cfa_offset -4\n")
    CFI(".cfi_restore %ecx\n")
    "popl    %edx\n"
    CFI(".cfi_adjust_cfa_offset -4\n")
    CFI(".cfi_restore %edx\n")
    "popl    %eax\n"
    CFI(".cfi_adjust_cfa_offset -4\n")
    CFI(".cfi_restore %eax\n")
    "popl    %ebp\n"
    CFI(".cfi_adjust_cfa_offset -4\n")
    CFI(".cfi_restore %ebp\n")
    "ret\n"
    CFI(".cfi_endproc\n")
    SIZE(X86CompilationCallback_SSE)
  );
# else
  // the following function is called only from this translation unit,
  // unless we are under 64bit Windows with MSC, where there is
  // no support for inline assembly
  static void X86CompilationCallback2(intptr_t *StackPtr, intptr_t RetAddr);

  _declspec(naked) void X86CompilationCallback(void) {
    __asm {
      push  ebp
      mov   ebp, esp
      push  eax
      push  edx
      push  ecx
      and   esp, -16
      sub   esp, 16
      mov   eax, dword ptr [ebp+4]
      mov   dword ptr [esp+4], eax
      mov   dword ptr [esp], ebp
      call  X86CompilationCallback2
      mov   esp, ebp
      sub   esp, 12
      pop   ecx
      pop   edx
      pop   eax
      pop   ebp
      ret
    }
  }

# endif // _MSC_VER

#else // Not an i386 host
  void X86CompilationCallback() {
    llvm_unreachable("Cannot call X86CompilationCallback() on a non-x86 arch!");
  }
#endif
}

/// X86CompilationCallback2 - This is the target-specific function invoked by the
/// function stub when we did not know the real target of a call.  This function
/// must locate the start of the stub or call site and pass it into the JIT
/// compiler function.
extern "C" {
#if !(defined (X86_64_JIT) && defined(_MSC_VER))
 // the following function is called only from this translation unit,
 // unless we are under 64bit Windows with MSC, where there is
 // no support for inline assembly
static
#endif
void LLVM_ATTRIBUTE_USED
X86CompilationCallback2(intptr_t *StackPtr, intptr_t RetAddr) {
  intptr_t *RetAddrLoc = &StackPtr[1];
  assert(*RetAddrLoc == RetAddr &&
         "Could not find return address on the stack!");

  // It's a stub if there is an interrupt marker after the call.
  bool isStub = ((unsigned char*)RetAddr)[0] == 0xCE;

  // The call instruction should have pushed the return value onto the stack...
#if defined (X86_64_JIT)
  RetAddr--;     // Backtrack to the reference itself...
#else
  RetAddr -= 4;  // Backtrack to the reference itself...
#endif

#if 0
  DEBUG(dbgs() << "In callback! Addr=" << (void*)RetAddr
               << " ESP=" << (void*)StackPtr
               << ": Resolving call to function: "
               << TheVM->getFunctionReferencedName((void*)RetAddr) << "\n");
#endif

  // Sanity check to make sure this really is a call instruction.
#if defined (X86_64_JIT)
  assert(((unsigned char*)RetAddr)[-2] == 0x41 &&"Not a call instr!");
  assert(((unsigned char*)RetAddr)[-1] == 0xFF &&"Not a call instr!");
#else
  assert(((unsigned char*)RetAddr)[-1] == 0xE8 &&"Not a call instr!");
#endif

  intptr_t NewVal = (intptr_t)JITCompilerFunction((void*)RetAddr);

  // Rewrite the call target... so that we don't end up here every time we
  // execute the call.
#if defined (X86_64_JIT)
  assert(isStub &&
         "X86-64 doesn't support rewriting non-stub lazy compilation calls:"
         " the call instruction varies too much.");
#else
  *(intptr_t *)RetAddr = (intptr_t)(NewVal-RetAddr-4);
#endif

  if (isStub) {
    // If this is a stub, rewrite the call into an unconditional branch
    // instruction so that two return addresses are not pushed onto the stack
    // when the requested function finally gets called.  This also makes the
    // 0xCE byte (interrupt) dead, so the marker doesn't effect anything.
#if defined (X86_64_JIT)
    // If the target address is within 32-bit range of the stub, use a
    // PC-relative branch instead of loading the actual address.  (This is
    // considerably shorter than the 64-bit immediate load already there.)
    // We assume here intptr_t is 64 bits.
    intptr_t diff = NewVal-RetAddr+7;
    if (diff >= -2147483648LL && diff <= 2147483647LL) {
      *(unsigned char*)(RetAddr-0xc) = 0xE9;
      *(intptr_t *)(RetAddr-0xb) = diff & 0xffffffff;
    } else {
      *(intptr_t *)(RetAddr - 0xa) = NewVal;
      ((unsigned char*)RetAddr)[0] = (2 | (4 << 3) | (3 << 6));
    }
    sys::ValgrindDiscardTranslations((void*)(RetAddr-0xc), 0xd);
#else
    ((unsigned char*)RetAddr)[-1] = 0xE9;
    sys::ValgrindDiscardTranslations((void*)(RetAddr-1), 5);
#endif
  }

  // Change the return address to reexecute the call instruction...
#if defined (X86_64_JIT)
  *RetAddrLoc -= 0xd;
#else
  *RetAddrLoc -= 5;
#endif
}
}

TargetJITInfo::LazyResolverFn
X86JITInfo::getLazyResolverFunction(JITCompilerFn F) {
  TsanIgnoreWritesBegin();
  JITCompilerFunction = F;
  TsanIgnoreWritesEnd();

#if defined (X86_32_JIT) && !defined (_MSC_VER)
  if (Subtarget->hasSSE1())
    return X86CompilationCallback_SSE;
#endif

  return X86CompilationCallback;
}

X86JITInfo::X86JITInfo(X86TargetMachine &tm) : TM(tm) {
  Subtarget = &TM.getSubtarget<X86Subtarget>();
  useGOT = 0;
  TLSOffset = 0;
}

void *X86JITInfo::emitGlobalValueIndirectSym(const GlobalValue* GV, void *ptr,
                                             JITCodeEmitter &JCE) {
#if defined (X86_64_JIT)
  const unsigned Alignment = 8;
  uint8_t Buffer[8];
  uint8_t *Cur = Buffer;
  MachineCodeEmitter::emitWordLEInto(Cur, (unsigned)(intptr_t)ptr);
  MachineCodeEmitter::emitWordLEInto(Cur, (unsigned)(((intptr_t)ptr) >> 32));
#else
  const unsigned Alignment = 4;
  uint8_t Buffer[4];
  uint8_t *Cur = Buffer;
  MachineCodeEmitter::emitWordLEInto(Cur, (intptr_t)ptr);
#endif
  return JCE.allocIndirectGV(GV, Buffer, sizeof(Buffer), Alignment);
}

TargetJITInfo::StubLayout X86JITInfo::getStubLayout() {
  // The 64-bit stub contains:
  //   movabs r10 <- 8-byte-target-address  # 10 bytes
  //   call|jmp *r10  # 3 bytes
  // The 32-bit stub contains a 5-byte call|jmp.
  // If the stub is a call to the compilation callback, an extra byte is added
  // to mark it as a stub.
  StubLayout Result = {14, 4};
  return Result;
}

void *X86JITInfo::emitFunctionStub(const Function* F, void *Target,
                                   JITCodeEmitter &JCE) {
  // Note, we cast to intptr_t here to silence a -pedantic warning that
  // complains about casting a function pointer to a normal pointer.
#if defined (X86_32_JIT) && !defined (_MSC_VER)
  bool NotCC = (Target != (void*)(intptr_t)X86CompilationCallback &&
                Target != (void*)(intptr_t)X86CompilationCallback_SSE);
#else
  bool NotCC = Target != (void*)(intptr_t)X86CompilationCallback;
#endif
  JCE.emitAlignment(4);
  void *Result = (void*)JCE.getCurrentPCValue();
  if (NotCC) {
#if defined (X86_64_JIT)
    JCE.emitByte(0x49);          // REX prefix
    JCE.emitByte(0xB8+2);        // movabsq r10
    JCE.emitWordLE((unsigned)(intptr_t)Target);
    JCE.emitWordLE((unsigned)(((intptr_t)Target) >> 32));
    JCE.emitByte(0x41);          // REX prefix
    JCE.emitByte(0xFF);          // jmpq *r10
    JCE.emitByte(2 | (4 << 3) | (3 << 6));
#else
    JCE.emitByte(0xE9);
    JCE.emitWordLE((intptr_t)Target-JCE.getCurrentPCValue()-4);
#endif
    return Result;
  }

#if defined (X86_64_JIT)
  JCE.emitByte(0x49);          // REX prefix
  JCE.emitByte(0xB8+2);        // movabsq r10
  JCE.emitWordLE((unsigned)(intptr_t)Target);
  JCE.emitWordLE((unsigned)(((intptr_t)Target) >> 32));
  JCE.emitByte(0x41);          // REX prefix
  JCE.emitByte(0xFF);          // callq *r10
  JCE.emitByte(2 | (2 << 3) | (3 << 6));
#else
  JCE.emitByte(0xE8);   // Call with 32 bit pc-rel destination...

  JCE.emitWordLE((intptr_t)Target-JCE.getCurrentPCValue()-4);
#endif

  // This used to use 0xCD, but that value is used by JITMemoryManager to
  // initialize the buffer with garbage, which means it may follow a
  // noreturn function call, confusing X86CompilationCallback2.  PR 4929.
  JCE.emitByte(0xCE);   // Interrupt - Just a marker identifying the stub!
  return Result;
}

/// getPICJumpTableEntry - Returns the value of the jumptable entry for the
/// specific basic block.
uintptr_t X86JITInfo::getPICJumpTableEntry(uintptr_t BB, uintptr_t Entry) {
#if defined(X86_64_JIT)
  return BB - Entry;
#else
  return BB - PICBase;
#endif
}

template<typename T> static void addUnaligned(void *Pos, T Delta) {
  T Value;
  std::memcpy(reinterpret_cast<char*>(&Value), reinterpret_cast<char*>(Pos),
              sizeof(T));
  Value += Delta;
  std::memcpy(reinterpret_cast<char*>(Pos), reinterpret_cast<char*>(&Value),
              sizeof(T));
}

/// relocate - Before the JIT can run a block of code that has been emitted,
/// it must rewrite the code to contain the actual addresses of any
/// referenced global symbols.
void X86JITInfo::relocate(void *Function, MachineRelocation *MR,
                          unsigned NumRelocs, unsigned char* GOTBase) {
  for (unsigned i = 0; i != NumRelocs; ++i, ++MR) {
    void *RelocPos = (char*)Function + MR->getMachineCodeOffset();
    intptr_t ResultPtr = (intptr_t)MR->getResultPointer();
    switch ((X86::RelocationType)MR->getRelocationType()) {
    case X86::reloc_pcrel_word: {
      // PC relative relocation, add the relocated value to the value already in
      // memory, after we adjust it for where the PC is.
      ResultPtr = ResultPtr -(intptr_t)RelocPos - 4 - MR->getConstantVal();
      addUnaligned<unsigned>(RelocPos, ResultPtr);
      break;
    }
    case X86::reloc_picrel_word: {
      // PIC base relative relocation, add the relocated value to the value
      // already in memory, after we adjust it for where the PIC base is.
      ResultPtr = ResultPtr - ((intptr_t)Function + MR->getConstantVal());
      addUnaligned<unsigned>(RelocPos, ResultPtr);
      break;
    }
    case X86::reloc_absolute_word:
    case X86::reloc_absolute_word_sext:
      // Absolute relocation, just add the relocated value to the value already
      // in memory.
      addUnaligned<unsigned>(RelocPos, ResultPtr);
      break;
    case X86::reloc_absolute_dword:
      addUnaligned<intptr_t>(RelocPos, ResultPtr);
      break;
    }
  }
}

char* X86JITInfo::allocateThreadLocalMemory(size_t size) {
#if defined(X86_32_JIT) && !defined(__APPLE__) && !defined(_MSC_VER)
  TLSOffset -= size;
  return TLSOffset;
#else
  llvm_unreachable("Cannot allocate thread local storage on this arch!");
#endif
}
