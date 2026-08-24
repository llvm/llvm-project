//===- MIRExport.cpp - Translate the MIR dialects back to .mir ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Registers the `export-mir` translation: rebuild an LLVM MachineFunction from
// a `mir`-dialect module and print it as .mir. Covers the generic subset
// (generic G_* opcodes, COPY, and ABI-boundary physical-register copies),
// enabling a .mir -> dialect -> .mir round trip.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MIR/IR/MIROps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Tools/mlir-translate/Translation.h"

#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/MIRPrinter.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/CodeGenTypes/LowLevelType.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"

using namespace mlir;

namespace {

/// Convert a `mir` dialect type to an LLT.
static llvm::LLT typeToLLT(Type t) {
  if (auto s = dyn_cast<mir::ScalarType>(t))
    return llvm::LLT::scalar(s.getWidth());
  if (auto p = dyn_cast<mir::PointerType>(t))
    return llvm::LLT::pointer(p.getAddressSpace(), p.getSizeInBits());
  if (auto v = dyn_cast<mir::VectorType>(t))
    return llvm::LLT::fixed_vector(v.getNumElements(),
                                   llvm::LLT::scalar(v.getElementType().getWidth()));
  return llvm::LLT();
}

/// Build a MachineFunction body from a mir.func and print nothing; the caller
/// prints via MIRPrinter.
class Exporter {
public:
  Exporter(llvm::MachineFunction &mf)
      : mf(mf), mri(mf.getRegInfo()),
        tii(mf.getSubtarget().getInstrInfo()),
        tri(mf.getSubtarget().getRegisterInfo()) {
    // Reverse map: opcode name -> opcode number.
    for (unsigned op = 0, e = tii->getNumOpcodes(); op != e; ++op)
      nameToOpcode[tii->getName(op)] = op;
  }

  LogicalResult run(mir::FuncOp func) {
    mbb = mf.CreateMachineBasicBlock();
    mf.push_back(mbb);
    for (Operation &op : func.getBody().front())
      if (failed(convert(&op)))
        return failure();
    return success();
  }

private:
  llvm::Register physRegByName(StringRef name) {
    for (unsigned r = 1, e = tri->getNumRegs(); r != e; ++r)
      if (name.equals_insensitive(tri->getName(r)))
        return r;
    return llvm::Register();
  }

  LogicalResult convert(Operation *op) {
    llvm::DebugLoc dl;
    if (auto fromPhys = dyn_cast<mir::CopyFromPhysOp>(op)) {
      llvm::Register phys = physRegByName(fromPhys.getReg().getName());
      if (!phys)
        return failure();
      llvm::Register dst =
          mri.createGenericVirtualRegister(typeToLLT(fromPhys.getType()));
      mbb->addLiveIn(phys);
      llvm::BuildMI(*mbb, mbb->end(), dl, tii->get(llvm::TargetOpcode::COPY),
                    dst)
          .addReg(phys);
      valMap[fromPhys.getResult()] = dst;
      return success();
    }
    if (auto toPhys = dyn_cast<mir::CopyToPhysOp>(op)) {
      llvm::Register phys = physRegByName(toPhys.getReg().getName());
      llvm::Register src = valMap.lookup(toPhys.getSrc());
      if (!phys || !src)
        return failure();
      llvm::BuildMI(*mbb, mbb->end(), dl, tii->get(llvm::TargetOpcode::COPY),
                    phys)
          .addReg(src);
      return success();
    }
    if (auto copy = dyn_cast<mir::CopyOp>(op)) {
      llvm::Register src = valMap.lookup(copy.getSrc());
      if (!src)
        return failure();
      llvm::Register dst =
          mri.createGenericVirtualRegister(typeToLLT(copy.getType()));
      llvm::BuildMI(*mbb, mbb->end(), dl, tii->get(llvm::TargetOpcode::COPY),
                    dst)
          .addReg(src);
      valMap[copy.getResult()] = dst;
      return success();
    }

    // Generic mir.g_* op -> corresponding generic opcode.
    StringRef opName = op->getName().stripDialect();
    if (opName.starts_with("g_")) {
      std::string upper = opName.upper(); // g_add -> G_ADD
      auto it = nameToOpcode.find(upper);
      if (it == nameToOpcode.end())
        return failure();
      auto mib = llvm::BuildMI(*mbb, mbb->end(), dl, tii->get(it->second));
      for (Value res : op->getResults()) {
        llvm::Register dst =
            mri.createGenericVirtualRegister(typeToLLT(res.getType()));
        mib.addDef(dst);
        valMap[res] = dst;
      }
      for (Value operand : op->getOperands()) {
        llvm::Register src = valMap.lookup(operand);
        if (!src)
          return failure();
        mib.addUse(src);
      }
      return success();
    }

    return failure();
  }

  llvm::MachineFunction &mf;
  llvm::MachineRegisterInfo &mri;
  const llvm::TargetInstrInfo *tii;
  const llvm::TargetRegisterInfo *tri;
  llvm::MachineBasicBlock *mbb = nullptr;
  llvm::StringMap<unsigned> nameToOpcode;
  DenseMap<Value, llvm::Register> valMap;
};

static LogicalResult translateModuleToMIR(Operation *op,
                                          llvm::raw_ostream &output) {
  auto module = dyn_cast<ModuleOp>(op);
  if (!module)
    return op->emitError("expected a module");

  LLVMInitializeAArch64TargetInfo();
  LLVMInitializeAArch64Target();
  LLVMInitializeAArch64TargetMC();

  llvm::LLVMContext llvmCtx;
  auto llvmModule = std::make_unique<llvm::Module>("mir", llvmCtx);
  llvm::Triple triple("aarch64-unknown-unknown");
  llvmModule->setTargetTriple(triple);

  std::string err;
  const llvm::Target *target = llvm::TargetRegistry::lookupTarget(triple, err);
  if (!target)
    return module.emitError("no AArch64 target: ") << err;
  llvm::TargetOptions options;
  std::unique_ptr<llvm::TargetMachine> tm(target->createTargetMachine(
      triple, "generic", "", options, /*RM=*/std::nullopt));
  llvmModule->setDataLayout(tm->createDataLayout());

  llvm::MachineModuleInfo mmi(tm.get());

  // Create an empty IR function + a MachineFunction for each mir.func.
  for (auto func : module.getOps<mir::FuncOp>()) {
    auto *fnTy = llvm::FunctionType::get(llvm::Type::getVoidTy(llvmCtx), false);
    auto *f = llvm::Function::Create(fnTy, llvm::GlobalValue::ExternalLinkage,
                                     func.getSymName(), llvmModule.get());
    llvm::BasicBlock *bb = llvm::BasicBlock::Create(llvmCtx, "entry", f);
    llvm::IRBuilder<>(bb).CreateRetVoid();

    llvm::MachineFunction &mf = mmi.getOrCreateMachineFunction(*f);
    if (failed(Exporter(mf).run(func)))
      return func.emitError("unsupported construct while exporting");
  }

  // Print the IR module then each machine function.
  llvm::printMIR(output, *llvmModule);
  for (llvm::Function &f : *llvmModule)
    if (llvm::MachineFunction *mf = mmi.getMachineFunction(f))
      llvm::printMIR(output, mmi, *mf);

  return success();
}

} // namespace

namespace mlir {
void registerToMIRTranslation() {
  TranslateFromMLIRRegistration registration(
      "export-mir", "Translate the MIR dialects to LLVM MachineIR (.mir)",
      translateModuleToMIR,
      [](DialectRegistry &registry) { registry.insert<mir::MIRDialect>(); });
}
} // namespace mlir
