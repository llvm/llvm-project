//===-- Assembler.cpp -------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Assembler.h"

#include "SnippetRepetitor.h"
#include "Target.h"
#include "llvm/CodeGen/GlobalISel/CallLowering.h"
#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/Support/MemoryBuffer.h"

namespace llvm {
namespace exegesis {

static constexpr const char ModuleID[] = "ExegesisInfoTest";
static constexpr const char FunctionID[] = "foo";

// Fills the given basic block with register setup code, and returns true if
// all registers could be setup correctly.
static bool
generateSnippetSetupCode(const ExegesisTarget &ET,
                         const llvm::MCSubtargetInfo *const MSI,
                         llvm::ArrayRef<RegisterValue> RegisterInitialValues,
                         BasicBlockFiller &BBF) {
  bool IsSnippetSetupComplete = true;
  for (const RegisterValue &RV : RegisterInitialValues) {
    // Load a constant in the register.
    const auto SetRegisterCode = ET.setRegTo(*MSI, RV.Register, RV.Value);
    if (SetRegisterCode.empty())
      IsSnippetSetupComplete = false;
    BBF.addInstructions(SetRegisterCode);
  }
  return IsSnippetSetupComplete;
}

// Small utility function to add named passes.
static bool addPass(llvm::PassManagerBase &PM, llvm::StringRef PassName,
                    llvm::TargetPassConfig &TPC) {
  const llvm::PassRegistry *PR = llvm::PassRegistry::getPassRegistry();
  const llvm::PassInfo *PI = PR->getPassInfo(PassName);
  if (!PI) {
    llvm::errs() << " run-pass " << PassName << " is not registered.\n";
    return true;
  }

  if (!PI->getNormalCtor()) {
    llvm::errs() << " cannot create pass: " << PI->getPassName() << "\n";
    return true;
  }
  llvm::Pass *P = PI->getNormalCtor()();
  std::string Banner = std::string("After ") + std::string(P->getPassName());
  PM.add(P);
  TPC.printAndVerify(Banner);

  return false;
}

llvm::MachineFunction &
createVoidVoidPtrMachineFunction(llvm::StringRef FunctionID,
                                 llvm::Module *Module,
                                 llvm::MachineModuleInfo *MMI) {
  llvm::Type *const ReturnType = llvm::Type::getInt32Ty(Module->getContext());
  llvm::Type *const MemParamType = llvm::PointerType::get(
      llvm::Type::getInt8Ty(Module->getContext()), 0 /*default address space*/);
  llvm::FunctionType *FunctionType =
      llvm::FunctionType::get(ReturnType, {MemParamType}, false);
  llvm::Function *const F = llvm::Function::Create(
      FunctionType, llvm::GlobalValue::InternalLinkage, FunctionID, Module);
  // Making sure we can create a MachineFunction out of this Function even if it
  // contains no IR.
  F->setIsMaterializable(true);
  return MMI->getOrCreateMachineFunction(*F);
}

BasicBlockFiller::BasicBlockFiller(llvm::MachineFunction &MF,
                                   llvm::MachineBasicBlock *MBB,
                                   const llvm::MCInstrInfo *MCII)
    : MF(MF), MBB(MBB), MCII(MCII) {}

void BasicBlockFiller::addInstruction(const llvm::MCInst &Inst,
                                      const llvm::DebugLoc &DL) {
  const unsigned Opcode = Inst.getOpcode();
  const llvm::MCInstrDesc &MCID = MCII->get(Opcode);
  llvm::MachineInstrBuilder Builder = llvm::BuildMI(MBB, DL, MCID);
  for (unsigned OpIndex = 0, E = Inst.getNumOperands(); OpIndex < E;
       ++OpIndex) {
    const llvm::MCOperand &Op = Inst.getOperand(OpIndex);
    if (Op.isReg()) {
      const bool IsDef = OpIndex < MCID.getNumDefs();
      unsigned Flags = 0;
      const llvm::MCOperandInfo &OpInfo = MCID.operands().begin()[OpIndex];
      if (IsDef && !OpInfo.isOptionalDef())
        Flags |= llvm::RegState::Define;
      Builder.addReg(Op.getReg(), Flags);
    } else if (Op.isImm()) {
      Builder.addImm(Op.getImm());
    } else if (!Op.isValid()) {
      llvm_unreachable("Operand is not set");
    } else {
      llvm_unreachable("Not yet implemented");
    }
  }
}

void BasicBlockFiller::addInstructions(ArrayRef<llvm::MCInst> Insts,
                                       const llvm::DebugLoc &DL) {
  for (const MCInst &Inst : Insts)
    addInstruction(Inst, DL);
}

void BasicBlockFiller::addReturn(const llvm::DebugLoc &DL) {
  // Insert the return code.
  const llvm::TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();
  if (TII->getReturnOpcode() < TII->getNumOpcodes()) {
    llvm::BuildMI(MBB, DL, TII->get(TII->getReturnOpcode()));
  } else {
    llvm::MachineIRBuilder MIB(MF);
    MIB.setMBB(*MBB);
    MF.getSubtarget().getCallLowering()->lowerReturn(MIB, nullptr, {});
  }
}

FunctionFiller::FunctionFiller(llvm::MachineFunction &MF,
                               std::vector<unsigned> RegistersSetUp)
    : MF(MF), MCII(MF.getTarget().getMCInstrInfo()), Entry(addBasicBlock()),
      RegistersSetUp(std::move(RegistersSetUp)) {}

BasicBlockFiller FunctionFiller::addBasicBlock() {
  llvm::MachineBasicBlock *MBB = MF.CreateMachineBasicBlock();
  MF.push_back(MBB);
  return BasicBlockFiller(MF, MBB, MCII);
}

ArrayRef<unsigned> FunctionFiller::getRegistersSetUp() const {
  return RegistersSetUp;
}

static std::unique_ptr<llvm::Module>
createModule(const std::unique_ptr<llvm::LLVMContext> &Context,
             const llvm::DataLayout DL) {
  auto Module = std::make_unique<llvm::Module>(ModuleID, *Context);
  Module->setDataLayout(DL);
  return Module;
}

llvm::BitVector getFunctionReservedRegs(const llvm::TargetMachine &TM) {
  std::unique_ptr<llvm::LLVMContext> Context =
      std::make_unique<llvm::LLVMContext>();
  std::unique_ptr<llvm::Module> Module =
      createModule(Context, TM.createDataLayout());
  // TODO: This only works for targets implementing LLVMTargetMachine.
  const LLVMTargetMachine &LLVMTM = static_cast<const LLVMTargetMachine &>(TM);
  std::unique_ptr<llvm::MachineModuleInfoWrapperPass> MMIWP =
      std::make_unique<llvm::MachineModuleInfoWrapperPass>(&LLVMTM);
  llvm::MachineFunction &MF = createVoidVoidPtrMachineFunction(
      FunctionID, Module.get(), &MMIWP.get()->getMMI());
  // Saving reserved registers for client.
  return MF.getSubtarget().getRegisterInfo()->getReservedRegs(MF);
}

void assembleToStream(const ExegesisTarget &ET,
                      std::unique_ptr<llvm::LLVMTargetMachine> TM,
                      llvm::ArrayRef<unsigned> LiveIns,
                      llvm::ArrayRef<RegisterValue> RegisterInitialValues,
                      const FillFunction &Fill,
                      llvm::raw_pwrite_stream &AsmStream) {
  std::unique_ptr<llvm::LLVMContext> Context =
      std::make_unique<llvm::LLVMContext>();
  std::unique_ptr<llvm::Module> Module =
      createModule(Context, TM->createDataLayout());
  std::unique_ptr<llvm::MachineModuleInfoWrapperPass> MMIWP =
      std::make_unique<llvm::MachineModuleInfoWrapperPass>(TM.get());
  llvm::MachineFunction &MF = createVoidVoidPtrMachineFunction(
      FunctionID, Module.get(), &MMIWP.get()->getMMI());

  // We need to instruct the passes that we're done with SSA and virtual
  // registers.
  auto &Properties = MF.getProperties();
  Properties.set(llvm::MachineFunctionProperties::Property::NoVRegs);
  Properties.reset(llvm::MachineFunctionProperties::Property::IsSSA);
  Properties.set(llvm::MachineFunctionProperties::Property::NoPHIs);

  for (const unsigned Reg : LiveIns)
    MF.getRegInfo().addLiveIn(Reg);

  std::vector<unsigned> RegistersSetUp;
  for (const auto &InitValue : RegisterInitialValues) {
    RegistersSetUp.push_back(InitValue.Register);
  }
  FunctionFiller Sink(MF, std::move(RegistersSetUp));
  auto Entry = Sink.getEntry();
  for (const unsigned Reg : LiveIns)
    Entry.MBB->addLiveIn(Reg);

  const bool IsSnippetSetupComplete = generateSnippetSetupCode(
      ET, TM->getMCSubtargetInfo(), RegisterInitialValues, Entry);

  // If the snippet setup is not complete, we disable liveliness tracking. This
  // means that we won't know what values are in the registers.
  if (!IsSnippetSetupComplete)
    Properties.reset(llvm::MachineFunctionProperties::Property::TracksLiveness);

  Fill(Sink);

  // prologue/epilogue pass needs the reserved registers to be frozen, this
  // is usually done by the SelectionDAGISel pass.
  MF.getRegInfo().freezeReservedRegs(MF);

  // We create the pass manager, run the passes to populate AsmBuffer.
  llvm::MCContext &MCContext = MMIWP->getMMI().getContext();
  llvm::legacy::PassManager PM;

  llvm::TargetLibraryInfoImpl TLII(llvm::Triple(Module->getTargetTriple()));
  PM.add(new llvm::TargetLibraryInfoWrapperPass(TLII));

  llvm::TargetPassConfig *TPC = TM->createPassConfig(PM);
  PM.add(TPC);
  PM.add(MMIWP.release());
  TPC->printAndVerify("MachineFunctionGenerator::assemble");
  // Add target-specific passes.
  ET.addTargetSpecificPasses(PM);
  TPC->printAndVerify("After ExegesisTarget::addTargetSpecificPasses");
  // Adding the following passes:
  // - machineverifier: checks that the MachineFunction is well formed.
  // - prologepilog: saves and restore callee saved registers.
  for (const char *PassName : {"machineverifier", "prologepilog"})
    if (addPass(PM, PassName, *TPC))
      llvm::report_fatal_error("Unable to add a mandatory pass");
  TPC->setInitialized();

  // AsmPrinter is responsible for generating the assembly into AsmBuffer.
  if (TM->addAsmPrinter(PM, AsmStream, nullptr,
                        llvm::TargetMachine::CGFT_ObjectFile, MCContext))
    llvm::report_fatal_error("Cannot add AsmPrinter passes");

  PM.run(*Module); // Run all the passes
}

llvm::object::OwningBinary<llvm::object::ObjectFile>
getObjectFromBuffer(llvm::StringRef InputData) {
  // Storing the generated assembly into a MemoryBuffer that owns the memory.
  std::unique_ptr<llvm::MemoryBuffer> Buffer =
      llvm::MemoryBuffer::getMemBufferCopy(InputData);
  // Create the ObjectFile from the MemoryBuffer.
  std::unique_ptr<llvm::object::ObjectFile> Obj = llvm::cantFail(
      llvm::object::ObjectFile::createObjectFile(Buffer->getMemBufferRef()));
  // Returning both the MemoryBuffer and the ObjectFile.
  return llvm::object::OwningBinary<llvm::object::ObjectFile>(
      std::move(Obj), std::move(Buffer));
}

llvm::object::OwningBinary<llvm::object::ObjectFile>
getObjectFromFile(llvm::StringRef Filename) {
  return llvm::cantFail(llvm::object::ObjectFile::createObjectFile(Filename));
}

namespace {

// Implementation of this class relies on the fact that a single object with a
// single function will be loaded into memory.
class TrackingSectionMemoryManager : public llvm::SectionMemoryManager {
public:
  explicit TrackingSectionMemoryManager(uintptr_t *CodeSize)
      : CodeSize(CodeSize) {}

  uint8_t *allocateCodeSection(uintptr_t Size, unsigned Alignment,
                               unsigned SectionID,
                               llvm::StringRef SectionName) override {
    *CodeSize = Size;
    return llvm::SectionMemoryManager::allocateCodeSection(
        Size, Alignment, SectionID, SectionName);
  }

private:
  uintptr_t *const CodeSize = nullptr;
};

} // namespace

ExecutableFunction::ExecutableFunction(
    std::unique_ptr<llvm::LLVMTargetMachine> TM,
    llvm::object::OwningBinary<llvm::object::ObjectFile> &&ObjectFileHolder)
    : Context(std::make_unique<llvm::LLVMContext>()) {
  assert(ObjectFileHolder.getBinary() && "cannot create object file");
  // Initializing the execution engine.
  // We need to use the JIT EngineKind to be able to add an object file.
  LLVMLinkInMCJIT();
  uintptr_t CodeSize = 0;
  std::string Error;
  ExecEngine.reset(
      llvm::EngineBuilder(createModule(Context, TM->createDataLayout()))
          .setErrorStr(&Error)
          .setMCPU(TM->getTargetCPU())
          .setEngineKind(llvm::EngineKind::JIT)
          .setMCJITMemoryManager(
              std::make_unique<TrackingSectionMemoryManager>(&CodeSize))
          .create(TM.release()));
  if (!ExecEngine)
    llvm::report_fatal_error(Error);
  // Adding the generated object file containing the assembled function.
  // The ExecutionEngine makes sure the object file is copied into an
  // executable page.
  ExecEngine->addObjectFile(std::move(ObjectFileHolder));
  // Fetching function bytes.
  FunctionBytes =
      llvm::StringRef(reinterpret_cast<const char *>(
                          ExecEngine->getFunctionAddress(FunctionID)),
                      CodeSize);
}

} // namespace exegesis
} // namespace llvm
