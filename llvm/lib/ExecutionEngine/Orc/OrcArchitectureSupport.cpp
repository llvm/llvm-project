//===------ OrcArchSupport.cpp - Architecture specific support code -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/Triple.h"
#include "llvm/ExecutionEngine/Orc/OrcArchitectureSupport.h"
#include "llvm/Support/Process.h"
#include <array>

namespace llvm {
namespace orc {

void OrcX86_64::writeResolverCode(uint8_t *ResolverMem, JITReentryFn ReentryFn,
                                  void *CallbackMgr) {

  const uint8_t ResolverCode[] = {
                                               // resolver_entry:
    0x55,                                      // 0x00: pushq     %rbp
    0x48, 0x89, 0xe5,                          // 0x01: movq      %rsp, %rbp
    0x50,                                      // 0x04: pushq     %rax
    0x53,                                      // 0x05: pushq     %rbx
    0x51,                                      // 0x06: pushq     %rcx
    0x52,                                      // 0x07: pushq     %rdx
    0x56,                                      // 0x08: pushq     %rsi
    0x57,                                      // 0x09: pushq     %rdi
    0x41, 0x50,                                // 0x0a: pushq     %r8
    0x41, 0x51,                                // 0x0c: pushq     %r9
    0x41, 0x52,                                // 0x0e: pushq     %r10
    0x41, 0x53,                                // 0x10: pushq     %r11
    0x41, 0x54,                                // 0x12: pushq     %r12
    0x41, 0x55,                                // 0x14: pushq     %r13
    0x41, 0x56,                                // 0x16: pushq     %r14
    0x41, 0x57,                                // 0x18: pushq     %r15
    0x48, 0x81, 0xec, 0x08, 0x02, 0x00, 0x00,  // 0x1a: subq      20, %rsp
    0x48, 0x0f, 0xae, 0x04, 0x24,              // 0x21: fxsave64  (%rsp)
    0x48, 0x8d, 0x3d, 0x43, 0x00, 0x00, 0x00,  // 0x26: leaq      67(%rip), %rdi
    0x48, 0x8b, 0x3f,                          // 0x2d: movq      (%rdi), %rdi
    0x48, 0x8b, 0x75, 0x08,                    // 0x30: movq      8(%rbp), %rsi
    0x48, 0x83, 0xee, 0x06,                    // 0x34: subq      $6, %rsi
    0x48, 0xb8,                                // 0x38: movabsq   $0, %rax

    // 0x3a: JIT re-entry fn addr:
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,

    0xff, 0xd0,                                // 0x42: callq     *%rax
    0x48, 0x89, 0x45, 0x08,                    // 0x44: movq      %rax, 8(%rbp)
    0x48, 0x0f, 0xae, 0x0c, 0x24,              // 0x48: fxrstor64 (%rsp)
    0x48, 0x81, 0xc4, 0x08, 0x02, 0x00, 0x00,  // 0x4d: addq      20, %rsp
    0x41, 0x5f,                                // 0x54: popq      %r15
    0x41, 0x5e,                                // 0x56: popq      %r14
    0x41, 0x5d,                                // 0x58: popq      %r13
    0x41, 0x5c,                                // 0x5a: popq      %r12
    0x41, 0x5b,                                // 0x5c: popq      %r11
    0x41, 0x5a,                                // 0x5e: popq      %r10
    0x41, 0x59,                                // 0x60: popq      %r9
    0x41, 0x58,                                // 0x62: popq      %r8
    0x5f,                                      // 0x64: popq      %rdi
    0x5e,                                      // 0x65: popq      %rsi
    0x5a,                                      // 0x66: popq      %rdx
    0x59,                                      // 0x67: popq      %rcx
    0x5b,                                      // 0x68: popq      %rbx
    0x58,                                      // 0x69: popq      %rax
    0x5d,                                      // 0x6a: popq      %rbp
    0xc3,                                      // 0x6b: retq
    0x00, 0x00, 0x00, 0x00,                    // 0x6c: <padding>

    // 0x70: Callback mgr address.
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  };

  const unsigned ReentryFnAddrOffset = 0x3a;
  const unsigned CallbackMgrAddrOffset = 0x70;
  
  memcpy(ResolverMem, ResolverCode, sizeof(ResolverCode));
  memcpy(ResolverMem + ReentryFnAddrOffset, &ReentryFn, sizeof(ReentryFn));
  memcpy(ResolverMem + CallbackMgrAddrOffset, &CallbackMgr,
         sizeof(CallbackMgr));
}

void OrcX86_64::writeTrampolines(uint8_t *TrampolineMem, void *ResolverAddr,
				 unsigned NumTrampolines) {

  unsigned OffsetToPtr = NumTrampolines * TrampolineSize;

  memcpy(TrampolineMem + OffsetToPtr, &ResolverAddr, sizeof(void*));

  uint64_t *Trampolines = reinterpret_cast<uint64_t*>(TrampolineMem);
  uint64_t CallIndirPCRel = 0xf1c40000000015ff;

  for (unsigned I = 0; I < NumTrampolines; ++I, OffsetToPtr -= TrampolineSize)
    Trampolines[I] = CallIndirPCRel | ((OffsetToPtr - 6) << 16);
}

std::error_code OrcX86_64::emitIndirectStubsBlock(IndirectStubsInfo &StubsInfo,
                                                  unsigned MinStubs,
                                                  void *InitialPtrVal) {
  // Stub format is:
  //
  // .section __orc_stubs
  // stub1:
  //                 jmpq    *ptr1(%rip)
  //                 .byte   0xC4         ; <- Invalid opcode padding.
  //                 .byte   0xF1
  // stub2:
  //                 jmpq    *ptr2(%rip)
  //
  // ...
  //
  // .section __orc_ptrs
  // ptr1:
  //                 .quad 0x0
  // ptr2:
  //                 .quad 0x0
  //
  // ...

  const unsigned StubSize = IndirectStubsInfo::StubSize;

  // Emit at least MinStubs, rounded up to fill the pages allocated.
  unsigned PageSize = sys::Process::getPageSize();
  unsigned NumPages = ((MinStubs * StubSize) + (PageSize - 1)) / PageSize;
  unsigned NumStubs = (NumPages * PageSize) / StubSize;

  // Allocate memory for stubs and pointers in one call.
  std::error_code EC;
  auto StubsMem =
    sys::OwningMemoryBlock(
      sys::Memory::allocateMappedMemory(2 * NumPages * PageSize, nullptr,
                                        sys::Memory::MF_READ |
                                        sys::Memory::MF_WRITE,
                                        EC));

  if (EC)
    return EC;

  // Create separate MemoryBlocks representing the stubs and pointers.
  sys::MemoryBlock StubsBlock(StubsMem.base(), NumPages * PageSize);
  sys::MemoryBlock PtrsBlock(static_cast<char*>(StubsMem.base()) +
                               NumPages * PageSize,
                             NumPages * PageSize);

  // Populate the stubs page stubs and mark it executable.
  uint64_t *Stub = reinterpret_cast<uint64_t*>(StubsBlock.base());
  uint64_t PtrOffsetField =
    static_cast<uint64_t>(NumPages * PageSize - 6) << 16;
  for (unsigned I = 0; I < NumStubs; ++I)
    Stub[I] = 0xF1C40000000025ff | PtrOffsetField;

  if (auto EC = sys::Memory::protectMappedMemory(StubsBlock,
                                                 sys::Memory::MF_READ |
                                                 sys::Memory::MF_EXEC))
    return EC;

  // Initialize all pointers to point at FailureAddress.
  void **Ptr = reinterpret_cast<void**>(PtrsBlock.base());
  for (unsigned I = 0; I < NumStubs; ++I)
    Ptr[I] = InitialPtrVal;

  StubsInfo = IndirectStubsInfo(NumStubs, std::move(StubsMem));

  return std::error_code();
}

} // End namespace orc.
} // End namespace llvm.
