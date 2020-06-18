//==- X86IndirectThunks.cpp - Construct indirect call/jump thunks for x86  --=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
///
/// Pass that injects an MI thunk that is used to lower indirect calls in a way
/// that prevents speculation on some x86 processors and can be used to mitigate
/// security vulnerabilities due to targeted speculative execution and side
/// channels such as CVE-2017-5715.
///
/// Currently supported thunks include:
/// - Retpoline -- A RET-implemented trampoline that lowers indirect calls
/// - LVI Thunk -- A CALL/JMP-implemented thunk that forces load serialization
///   before making an indirect call/jump
///
/// Note that the reason that this is implemented as a MachineFunctionPass and
/// not a ModulePass is that ModulePasses at this point in the LLVM X86 pipeline
/// serialize all transformations, which can consume lots of memory.
///
/// TODO(chandlerc): All of this code could use better comments and
/// documentation.
///
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrBuilder.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/IndirectThunks.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"

using namespace llvm;

#define DEBUG_TYPE "x86-retpoline-thunks"

static const char RetpolineNamePrefix[] = "__llvm_retpoline_";
static const char R11RetpolineName[] = "__llvm_retpoline_r11";
static const char EAXRetpolineName[] = "__llvm_retpoline_eax";
static const char ECXRetpolineName[] = "__llvm_retpoline_ecx";
static const char EDXRetpolineName[] = "__llvm_retpoline_edx";
static const char EDIRetpolineName[] = "__llvm_retpoline_edi";

static const char LVIThunkNamePrefix[] = "__llvm_lvi_thunk_";
static const char R11LVIThunkName[] = "__llvm_lvi_thunk_r11";

namespace {
struct RetpolineThunkInserter : ThunkInserter<RetpolineThunkInserter> {
  const char *getThunkPrefix() { return RetpolineNamePrefix; }
  bool mayUseThunk(const MachineFunction &MF) {
    const auto &STI = MF.getSubtarget<X86Subtarget>();
    return (STI.useRetpolineIndirectCalls() ||
            STI.useRetpolineIndirectBranches()) &&
           !STI.useRetpolineExternalThunk();
  }
  void insertThunks(MachineModuleInfo &MMI);
  void populateThunk(MachineFunction &MF);
};

struct LVIThunkInserter : ThunkInserter<LVIThunkInserter> {
  const char *getThunkPrefix() { return LVIThunkNamePrefix; }
  bool mayUseThunk(const MachineFunction &MF) {
    return MF.getSubtarget<X86Subtarget>().useLVIControlFlowIntegrity();
  }
  void insertThunks(MachineModuleInfo &MMI) {
    createThunkFunction(MMI, R11LVIThunkName);
  }
  void populateThunk(MachineFunction &MF) {
    assert (MF.size() == 1);
    MachineBasicBlock *Entry = &MF.front();
    Entry->clear();

    // This code mitigates LVI by replacing each indirect call/jump with a
    // direct call/jump to a thunk that looks like:
    // ```
    // lfence
    // jmpq *%r11
    // ```
    // This ensures that if the value in register %r11 was loaded from memory,
    // then the value in %r11 is (architecturally) correct prior to the jump.
    const TargetInstrInfo *TII = MF.getSubtarget<X86Subtarget>().getInstrInfo();
    BuildMI(&MF.front(), DebugLoc(), TII->get(X86::LFENCE));
    BuildMI(&MF.front(), DebugLoc(), TII->get(X86::JMP64r)).addReg(X86::R11);
    MF.front().addLiveIn(X86::R11);
    return;
  }
};

class X86IndirectThunks : public MachineFunctionPass {
public:
  static char ID;

  X86IndirectThunks() : MachineFunctionPass(ID) {}

  StringRef getPassName() const override { return "X86 Indirect Thunks"; }

  bool doInitialization(Module &M) override;
  bool runOnMachineFunction(MachineFunction &MF) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    MachineFunctionPass::getAnalysisUsage(AU);
    AU.addRequired<MachineModuleInfoWrapperPass>();
    AU.addPreserved<MachineModuleInfoWrapperPass>();
  }

private:
  std::tuple<RetpolineThunkInserter, LVIThunkInserter> TIs;

  // FIXME: When LLVM moves to C++17, these can become folds
  template <typename... ThunkInserterT>
  static void initTIs(Module &M,
                      std::tuple<ThunkInserterT...> &ThunkInserters) {
    (void)std::initializer_list<int>{
        (std::get<ThunkInserterT>(ThunkInserters).init(M), 0)...};
  }
  template <typename... ThunkInserterT>
  static bool runTIs(MachineModuleInfo &MMI, MachineFunction &MF,
                     std::tuple<ThunkInserterT...> &ThunkInserters) {
    bool Modified = false;
    (void)std::initializer_list<int>{
        Modified |= std::get<ThunkInserterT>(ThunkInserters).run(MMI, MF)...};
    return Modified;
  }
};

} // end anonymous namespace

void RetpolineThunkInserter::insertThunks(MachineModuleInfo &MMI) {
  if (MMI.getTarget().getTargetTriple().getArch() == Triple::x86_64)
    createThunkFunction(MMI, R11RetpolineName);
  else
    for (StringRef Name : {EAXRetpolineName, ECXRetpolineName, EDXRetpolineName,
                           EDIRetpolineName})
      createThunkFunction(MMI, Name);
}

void RetpolineThunkInserter::populateThunk(MachineFunction &MF) {
  bool Is64Bit = MF.getTarget().getTargetTriple().getArch() == Triple::x86_64;
  Register ThunkReg;
  if (Is64Bit) {
    assert(MF.getName() == "__llvm_retpoline_r11" &&
           "Should only have an r11 thunk on 64-bit targets");

    // __llvm_retpoline_r11:
    //   callq .Lr11_call_target
    // .Lr11_capture_spec:
    //   pause
    //   lfence
    //   jmp .Lr11_capture_spec
    // .align 16
    // .Lr11_call_target:
    //   movq %r11, (%rsp)
    //   retq
    ThunkReg = X86::R11;
  } else {
    // For 32-bit targets we need to emit a collection of thunks for various
    // possible scratch registers as well as a fallback that uses EDI, which is
    // normally callee saved.
    //   __llvm_retpoline_eax:
    //         calll .Leax_call_target
    //   .Leax_capture_spec:
    //         pause
    //         jmp .Leax_capture_spec
    //   .align 16
    //   .Leax_call_target:
    //         movl %eax, (%esp)  # Clobber return addr
    //         retl
    //
    //   __llvm_retpoline_ecx:
    //   ... # Same setup
    //         movl %ecx, (%esp)
    //         retl
    //
    //   __llvm_retpoline_edx:
    //   ... # Same setup
    //         movl %edx, (%esp)
    //         retl
    //
    //   __llvm_retpoline_edi:
    //   ... # Same setup
    //         movl %edi, (%esp)
    //         retl
    if (MF.getName() == EAXRetpolineName)
      ThunkReg = X86::EAX;
    else if (MF.getName() == ECXRetpolineName)
      ThunkReg = X86::ECX;
    else if (MF.getName() == EDXRetpolineName)
      ThunkReg = X86::EDX;
    else if (MF.getName() == EDIRetpolineName)
      ThunkReg = X86::EDI;
    else
      llvm_unreachable("Invalid thunk name on x86-32!");
  }

  const TargetInstrInfo *TII = MF.getSubtarget<X86Subtarget>().getInstrInfo();
  assert (MF.size() == 1);
  MachineBasicBlock *Entry = &MF.front();
  Entry->clear();

  MachineBasicBlock *CaptureSpec =
      MF.CreateMachineBasicBlock(Entry->getBasicBlock());
  MachineBasicBlock *CallTarget =
      MF.CreateMachineBasicBlock(Entry->getBasicBlock());
  MCSymbol *TargetSym = MF.getContext().createTempSymbol();
  MF.push_back(CaptureSpec);
  MF.push_back(CallTarget);

  const unsigned CallOpc = Is64Bit ? X86::CALL64pcrel32 : X86::CALLpcrel32;
  const unsigned RetOpc = Is64Bit ? X86::RETQ : X86::RETL;

  Entry->addLiveIn(ThunkReg);
  BuildMI(Entry, DebugLoc(), TII->get(CallOpc)).addSym(TargetSym);

  // The MIR verifier thinks that the CALL in the entry block will fall through
  // to CaptureSpec, so mark it as the successor. Technically, CaptureTarget is
  // the successor, but the MIR verifier doesn't know how to cope with that.
  Entry->addSuccessor(CaptureSpec);

  // In the capture loop for speculation, we want to stop the processor from
  // speculating as fast as possible. On Intel processors, the PAUSE instruction
  // will block speculation without consuming any execution resources. On AMD
  // processors, the PAUSE instruction is (essentially) a nop, so we also use an
  // LFENCE instruction which they have advised will stop speculation as well
  // with minimal resource utilization. We still end the capture with a jump to
  // form an infinite loop to fully guarantee that no matter what implementation
  // of the x86 ISA, speculating this code path never escapes.
  BuildMI(CaptureSpec, DebugLoc(), TII->get(X86::PAUSE));
  BuildMI(CaptureSpec, DebugLoc(), TII->get(X86::LFENCE));
  BuildMI(CaptureSpec, DebugLoc(), TII->get(X86::JMP_1)).addMBB(CaptureSpec);
  CaptureSpec->setHasAddressTaken();
  CaptureSpec->addSuccessor(CaptureSpec);

  CallTarget->addLiveIn(ThunkReg);
  CallTarget->setHasAddressTaken();
  CallTarget->setAlignment(Align(16));

  // Insert return address clobber
  const unsigned MovOpc = Is64Bit ? X86::MOV64mr : X86::MOV32mr;
  const Register SPReg = Is64Bit ? X86::RSP : X86::ESP;
  addRegOffset(BuildMI(CallTarget, DebugLoc(), TII->get(MovOpc)), SPReg, false,
               0)
      .addReg(ThunkReg);

  CallTarget->back().setPreInstrSymbol(MF, TargetSym);
  BuildMI(CallTarget, DebugLoc(), TII->get(RetOpc));
}

FunctionPass *llvm::createX86IndirectThunksPass() {
  return new X86IndirectThunks();
}

char X86IndirectThunks::ID = 0;

bool X86IndirectThunks::doInitialization(Module &M) {
  initTIs(M, TIs);
  return false;
}

bool X86IndirectThunks::runOnMachineFunction(MachineFunction &MF) {
  LLVM_DEBUG(dbgs() << getPassName() << '\n');
  auto &MMI = getAnalysis<MachineModuleInfoWrapperPass>().getMMI();
  return runTIs(MMI, MF, TIs);
}
