//======- X86RetpolineThunks.cpp - Construct retpoline thunks for x86  --=====//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
///
/// Pass that injects an MI thunk implementing a "retpoline". This is
/// a RET-implemented trampoline that is used to lower indirect calls in a way
/// that prevents speculation on some x86 processors and can be used to mitigate
/// security vulnerabilities due to targeted speculative execution and side
/// channels such as CVE-2017-5715.
///
/// TODO(chandlerc): All of this code could use better comments and
/// documentation.
///
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrBuilder.h"
#include "X86Subtarget.h"
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

using namespace llvm;

#define DEBUG_TYPE "x86-retpoline-thunks"

namespace {
class X86RetpolineThunks : public ModulePass {
public:
  static char ID;

  X86RetpolineThunks() : ModulePass(ID) {}

  StringRef getPassName() const override { return "X86 Retpoline Thunks"; }

  bool runOnModule(Module &M) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<MachineModuleInfo>();
    AU.addPreserved<MachineModuleInfo>();
  }

private:
  MachineModuleInfo *MMI;
  const TargetMachine *TM;
  bool Is64Bit;
  const X86Subtarget *STI;
  const X86InstrInfo *TII;

  Function *createThunkFunction(Module &M, StringRef Name);
  void insertRegReturnAddrClobber(MachineBasicBlock &MBB, unsigned Reg);
  void insert32BitPushReturnAddrClobber(MachineBasicBlock &MBB);
  void createThunk(Module &M, StringRef NameSuffix,
                   Optional<unsigned> Reg = None);
};

} // end anonymous namespace

ModulePass *llvm::createX86RetpolineThunksPass() {
  return new X86RetpolineThunks();
}

char X86RetpolineThunks::ID = 0;

bool X86RetpolineThunks::runOnModule(Module &M) {
  DEBUG(dbgs() << getPassName() << '\n');

  auto *TPC = getAnalysisIfAvailable<TargetPassConfig>();
  assert(TPC && "X86-specific target pass should not be run without a target "
                "pass config!");

  MMI = &getAnalysis<MachineModuleInfo>();
  TM = &TPC->getTM<TargetMachine>();
  Is64Bit = TM->getTargetTriple().getArch() == Triple::x86_64;

  // Only add a thunk if we have at least one function that has the retpoline
  // feature enabled in its subtarget.
  // FIXME: Conditionalize on indirect calls so we don't emit a thunk when
  // nothing will end up calling it.
  // FIXME: It's a little silly to look at every function just to enumerate
  // the subtargets, but eventually we'll want to look at them for indirect
  // calls, so maybe this is OK.
  if (!llvm::any_of(M, [&](const Function &F) {
        // Save the subtarget we find for use in emitting the subsequent
        // thunk.
        STI = &TM->getSubtarget<X86Subtarget>(F);
        return STI->useRetpoline() && !STI->useRetpolineExternalThunk();
      }))
    return false;

  // If we have a relevant subtarget, get the instr info as well.
  TII = STI->getInstrInfo();

  if (Is64Bit) {
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

    createThunk(M, "r11", X86::R11);
  } else {
    // For 32-bit targets we need to emit a collection of thunks for various
    // possible scratch registers as well as a fallback that is used when
    // there are no scratch registers and assumes the retpoline target has
    // been pushed.
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
    // This last one is a bit more special and so needs a little extra
    // handling.
    // __llvm_retpoline_push:
    //         calll .Lpush_call_target
    // .Lpush_capture_spec:
    //         pause
    //         lfence
    //         jmp .Lpush_capture_spec
    // .align 16
    // .Lpush_call_target:
    //         # Clear pause_loop return address.
    //         addl $4, %esp
    //         # Top of stack words are: Callee, RA. Exchange Callee and RA.
    //         pushl 4(%esp)  # Push callee
    //         pushl 4(%esp)  # Push RA
    //         popl 8(%esp)   # Pop RA to final RA
    //         popl (%esp)    # Pop callee to next top of stack
    //         retl           # Ret to callee
    createThunk(M, "eax", X86::EAX);
    createThunk(M, "ecx", X86::ECX);
    createThunk(M, "edx", X86::EDX);
    createThunk(M, "push");
  }

  return true;
}

Function *X86RetpolineThunks::createThunkFunction(Module &M, StringRef Name) {
  LLVMContext &Ctx = M.getContext();
  auto Type = FunctionType::get(Type::getVoidTy(Ctx), false);
  Function *F =
      Function::Create(Type, GlobalValue::LinkOnceODRLinkage, Name, &M);
  F->setVisibility(GlobalValue::HiddenVisibility);
  F->setComdat(M.getOrInsertComdat(Name));

  // Add Attributes so that we don't create a frame, unwind information, or
  // inline.
  AttrBuilder B;
  B.addAttribute(llvm::Attribute::NoUnwind);
  B.addAttribute(llvm::Attribute::Naked);
  F->addAttributes(llvm::AttributeList::FunctionIndex, B);

  // Populate our function a bit so that we can verify.
  BasicBlock *Entry = BasicBlock::Create(Ctx, "entry", F);
  IRBuilder<> Builder(Entry);

  Builder.CreateRetVoid();
  return F;
}

void X86RetpolineThunks::insertRegReturnAddrClobber(MachineBasicBlock &MBB,
                                                    unsigned Reg) {
  const unsigned MovOpc = Is64Bit ? X86::MOV64mr : X86::MOV32mr;
  const unsigned SPReg = Is64Bit ? X86::RSP : X86::ESP;
  addRegOffset(BuildMI(&MBB, DebugLoc(), TII->get(MovOpc)), SPReg, false, 0)
      .addReg(Reg);
}
void X86RetpolineThunks::insert32BitPushReturnAddrClobber(
    MachineBasicBlock &MBB) {
  // The instruction sequence we use to replace the return address without
  // a scratch register is somewhat complicated:
  //   # Clear capture_spec from return address.
  //   addl $4, %esp
  //   # Top of stack words are: Callee, RA. Exchange Callee and RA.
  //   pushl 4(%esp)  # Push callee
  //   pushl 4(%esp)  # Push RA
  //   popl 8(%esp)   # Pop RA to final RA
  //   popl (%esp)    # Pop callee to next top of stack
  //   retl           # Ret to callee
  BuildMI(&MBB, DebugLoc(), TII->get(X86::ADD32ri), X86::ESP)
      .addReg(X86::ESP)
      .addImm(4);
  addRegOffset(BuildMI(&MBB, DebugLoc(), TII->get(X86::PUSH32rmm)), X86::ESP,
               false, 4);
  addRegOffset(BuildMI(&MBB, DebugLoc(), TII->get(X86::PUSH32rmm)), X86::ESP,
               false, 4);
  addRegOffset(BuildMI(&MBB, DebugLoc(), TII->get(X86::POP32rmm)), X86::ESP,
               false, 8);
  addRegOffset(BuildMI(&MBB, DebugLoc(), TII->get(X86::POP32rmm)), X86::ESP,
               false, 0);
}

void X86RetpolineThunks::createThunk(Module &M, StringRef NameSuffix,
                                     Optional<unsigned> Reg) {
  Function &F =
      *createThunkFunction(M, (Twine("__llvm_retpoline_") + NameSuffix).str());
  MachineFunction &MF = MMI->getOrCreateMachineFunction(F);

  // Set MF properties. We never use vregs...
  MF.getProperties().set(MachineFunctionProperties::Property::NoVRegs);

  BasicBlock &OrigEntryBB = F.getEntryBlock();
  MachineBasicBlock *Entry = MF.CreateMachineBasicBlock(&OrigEntryBB);
  MachineBasicBlock *CaptureSpec = MF.CreateMachineBasicBlock(&OrigEntryBB);
  MachineBasicBlock *CallTarget = MF.CreateMachineBasicBlock(&OrigEntryBB);

  MF.push_back(Entry);
  MF.push_back(CaptureSpec);
  MF.push_back(CallTarget);

  const unsigned CallOpc = Is64Bit ? X86::CALL64pcrel32 : X86::CALLpcrel32;
  const unsigned RetOpc = Is64Bit ? X86::RETQ : X86::RETL;

  BuildMI(Entry, DebugLoc(), TII->get(CallOpc)).addMBB(CallTarget);
  Entry->addSuccessor(CallTarget);
  Entry->addSuccessor(CaptureSpec);
  CallTarget->setHasAddressTaken();

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

  CallTarget->setAlignment(4);
  if (Reg) {
    insertRegReturnAddrClobber(*CallTarget, *Reg);
  } else {
    assert(!Is64Bit && "We only support non-reg thunks on 32-bit x86!");
    insert32BitPushReturnAddrClobber(*CallTarget);
  }
  BuildMI(CallTarget, DebugLoc(), TII->get(RetOpc));
}
