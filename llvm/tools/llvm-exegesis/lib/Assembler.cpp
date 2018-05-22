//===-- Assembler.cpp -------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Assembler.h"

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

namespace exegesis {

static constexpr const char ModuleID[] = "ExegesisInfoTest";
static constexpr const char FunctionID[] = "foo";

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

// Creates a void MachineFunction with no argument.
static llvm::MachineFunction &
createVoidVoidMachineFunction(llvm::StringRef FunctionID, llvm::Module *Module,
                              llvm::MachineModuleInfo *MMI) {
  llvm::Type *const ReturnType = llvm::Type::getInt32Ty(Module->getContext());
  llvm::FunctionType *FunctionType = llvm::FunctionType::get(ReturnType, false);
  llvm::Function *const F = llvm::Function::Create(
      FunctionType, llvm::GlobalValue::InternalLinkage, FunctionID, Module);
  // Making sure we can create a MachineFunction out of this Function even if it
  // contains no IR.
  F->setIsMaterializable(true);
  return MMI->getOrCreateMachineFunction(*F);
}

static void fillMachineFunction(llvm::MachineFunction &MF,
                                llvm::ArrayRef<llvm::MCInst> Instructions) {
  llvm::MachineBasicBlock *MBB = MF.CreateMachineBasicBlock();
  MF.push_back(MBB);
  const llvm::MCInstrInfo *MCII = MF.getTarget().getMCInstrInfo();
  llvm::DebugLoc DL;
  for (const llvm::MCInst &Inst : Instructions) {
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
      } else {
        llvm_unreachable("Not yet implemented");
      }
    }
  }
  // Insert the return code.
  const llvm::TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();
  if (TII->getReturnOpcode() < TII->getNumOpcodes()) {
    llvm::BuildMI(MBB, DL, TII->get(TII->getReturnOpcode()));
  } else {
    llvm::MachineIRBuilder MIB(MF);
    MIB.setMBB(*MBB);
    MF.getSubtarget().getCallLowering()->lowerReturn(MIB, nullptr, 0);
  }
}

static std::unique_ptr<llvm::Module>
createModule(const std::unique_ptr<llvm::LLVMContext> &Context,
             const llvm::DataLayout DL) {
  auto Module = llvm::make_unique<llvm::Module>(ModuleID, *Context);
  Module->setDataLayout(DL);
  return Module;
}

llvm::BitVector getFunctionReservedRegs(const llvm::TargetMachine &TM) {
  std::unique_ptr<llvm::LLVMContext> Context =
      llvm::make_unique<llvm::LLVMContext>();
  std::unique_ptr<llvm::Module> Module =
      createModule(Context, TM.createDataLayout());
  std::unique_ptr<llvm::MachineModuleInfo> MMI =
      llvm::make_unique<llvm::MachineModuleInfo>(&TM);
  llvm::MachineFunction &MF =
      createVoidVoidMachineFunction(FunctionID, Module.get(), MMI.get());
  // Saving reserved registers for client.
  return MF.getSubtarget().getRegisterInfo()->getReservedRegs(MF);
}

void assembleToStream(std::unique_ptr<llvm::LLVMTargetMachine> TM,
                      llvm::ArrayRef<llvm::MCInst> Instructions,
                      llvm::raw_pwrite_stream &AsmStream) {
  std::unique_ptr<llvm::LLVMContext> Context =
      llvm::make_unique<llvm::LLVMContext>();
  std::unique_ptr<llvm::Module> Module =
      createModule(Context, TM->createDataLayout());
  std::unique_ptr<llvm::MachineModuleInfo> MMI =
      llvm::make_unique<llvm::MachineModuleInfo>(TM.get());
  llvm::MachineFunction &MF =
      createVoidVoidMachineFunction(FunctionID, Module.get(), MMI.get());

  // We need to instruct the passes that we're done with SSA and virtual
  // registers.
  auto &Properties = MF.getProperties();
  Properties.set(llvm::MachineFunctionProperties::Property::NoVRegs);
  Properties.reset(llvm::MachineFunctionProperties::Property::IsSSA);
  Properties.reset(llvm::MachineFunctionProperties::Property::TracksLiveness);
  // prologue/epilogue pass needs the reserved registers to be frozen, this
  // is usually done by the SelectionDAGISel pass.
  MF.getRegInfo().freezeReservedRegs(MF);

  // Fill the MachineFunction from the instructions.
  fillMachineFunction(MF, Instructions);

  // We create the pass manager, run the passes to populate AsmBuffer.
  llvm::MCContext &MCContext = MMI->getContext();
  llvm::legacy::PassManager PM;

  llvm::TargetLibraryInfoImpl TLII(llvm::Triple(Module->getTargetTriple()));
  PM.add(new llvm::TargetLibraryInfoWrapperPass(TLII));

  llvm::TargetPassConfig *TPC = TM->createPassConfig(PM);
  PM.add(TPC);
  PM.add(MMI.release());
  TPC->printAndVerify("MachineFunctionGenerator::assemble");
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
    : Context(llvm::make_unique<llvm::LLVMContext>()) {
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
              llvm::make_unique<TrackingSectionMemoryManager>(&CodeSize))
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
