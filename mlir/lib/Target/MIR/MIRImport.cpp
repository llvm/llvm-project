//===- MIRImport.cpp - Translate LLVM .mir into the MIR dialects ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Registers the `import-mir` translation: parse a pre-register-allocation
// LLVM MachineIR (.mir) file and build the equivalent `mir` (+ `aarch64_mir`)
// dialect module. This first cut targets the generic (post-legalize /
// post-regbankselect) subset: generic G_* opcodes, COPY, and the ABI-boundary
// physical-register copies. Selected target opcodes are handled in a later
// step.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MIR/IR/MIROps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Tools/mlir-translate/Translation.h"

#include "llvm/CodeGen/MIRParser/MIRParser.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/CodeGenTypes/LowLevelType.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"

using namespace mlir;

namespace {

/// Convert an LLT to the corresponding `mir` dialect type.
static Type lltToType(llvm::LLT llt, MLIRContext *ctx) {
  if (llt.isPointer())
    return mir::PointerType::get(ctx, llt.getAddressSpace(),
                                 llt.getSizeInBits());
  if (llt.isVector()) {
    auto elt = cast<mir::ScalarType>(lltToType(llt.getElementType(), ctx));
    return mir::VectorType::get(ctx, llt.getElementCount().getFixedValue(),
                                elt);
  }
  // Treat everything else as a scalar of the given bit width.
  return mir::ScalarType::get(ctx, llt.getSizeInBits());
}

/// Importer state for a single translation.
class Importer {
public:
  Importer(MLIRContext *ctx, const llvm::MachineFunction &mf)
      : ctx(ctx), mf(mf), mri(mf.getRegInfo()),
        tii(mf.getSubtarget().getInstrInfo()),
        tri(mf.getSubtarget().getRegisterInfo()) {}

  /// Build the body of a mir.func for `mf` into `block`. Returns failure on an
  /// unsupported construct.
  LogicalResult run(Block *block) {
    OpBuilder b(block, block->end());
    Location loc = UnknownLoc::get(ctx);
    for (const llvm::MachineBasicBlock &mbb : mf)
      for (const llvm::MachineInstr &mi : mbb)
        if (failed(convert(b, loc, mi)))
          return failure();
    return success();
  }

private:
  Value lookup(llvm::Register reg) { return vregMap.lookup(reg); }
  void map(llvm::Register reg, Value v) { vregMap[reg] = v; }

  StringRef physRegName(llvm::Register reg) { return tri->getName(reg); }

  LogicalResult convert(OpBuilder &b, Location loc,
                        const llvm::MachineInstr &mi) {
    unsigned opcode = mi.getOpcode();

    // COPY: virtual<->physical boundary copies and virtual->virtual copies.
    if (opcode == llvm::TargetOpcode::COPY) {
      const llvm::MachineOperand &dst = mi.getOperand(0);
      const llvm::MachineOperand &src = mi.getOperand(1);
      // vreg = COPY $phys  -> mir.copy_from_phys
      if (dst.getReg().isVirtual() && src.getReg().isPhysical()) {
        Type ty = lltToType(mri.getType(dst.getReg()), ctx);
        auto reg = mir::PhysRegAttr::get(ctx, physRegName(src.getReg()));
        auto op = b.create<mir::CopyFromPhysOp>(loc, ty, reg);
        map(dst.getReg(), op.getResult());
        return success();
      }
      // $phys = COPY vreg  -> mir.copy_to_phys
      if (dst.getReg().isPhysical() && src.getReg().isVirtual()) {
        Value v = lookup(src.getReg());
        if (!v)
          return failure();
        auto reg = mir::PhysRegAttr::get(ctx, physRegName(dst.getReg()));
        b.create<mir::CopyToPhysOp>(loc, v, reg);
        return success();
      }
      // vreg = COPY vreg   -> mir.copy
      if (dst.getReg().isVirtual() && src.getReg().isVirtual()) {
        Value v = lookup(src.getReg());
        if (!v)
          return failure();
        auto op = b.create<mir::CopyOp>(loc, v.getType(), v);
        map(dst.getReg(), op.getResult());
        return success();
      }
      return failure();
    }

    // Generic (pre-isel) opcodes -> mir.g_<lowercase name>.
    if (llvm::isPreISelGenericOpcode(opcode)) {
      std::string name = ("mir." + tii->getName(opcode)).str();
      for (char &c : name)
        c = llvm::toLower(c);

      SmallVector<Value> operands;
      SmallVector<Type> resultTypes;
      SmallVector<llvm::Register> defRegs;
      for (const llvm::MachineOperand &mo : mi.operands()) {
        if (!mo.isReg())
          continue; // immediates handled below
        if (mo.isImplicit())
          continue;
        if (mo.isDef()) {
          resultTypes.push_back(lltToType(mri.getType(mo.getReg()), ctx));
          defRegs.push_back(mo.getReg());
        } else {
          Value v = lookup(mo.getReg());
          if (!v)
            return failure();
          operands.push_back(v);
        }
      }

      OperationState state(loc, name);
      state.addOperands(operands);
      state.addTypes(resultTypes);
      Operation *op = b.create(state);
      for (auto [reg, res] : llvm::zip(defRegs, op->getResults()))
        map(reg, res);
      return success();
    }

    // Selected target opcodes are handled in a later step.
    return failure();
  }

  MLIRContext *ctx;
  const llvm::MachineFunction &mf;
  const llvm::MachineRegisterInfo &mri;
  const llvm::TargetInstrInfo *tii;
  const llvm::TargetRegisterInfo *tri;
  DenseMap<llvm::Register, Value> vregMap;
};

} // namespace

/// Parse a .mir buffer and build a `mir`-dialect module.
static OwningOpRef<Operation *> translateMIRToModule(llvm::SourceMgr &sourceMgr,
                                                     MLIRContext *ctx) {
  // Only AArch64 is needed for these dialects.
  LLVMInitializeAArch64TargetInfo();
  LLVMInitializeAArch64Target();
  LLVMInitializeAArch64TargetMC();

  llvm::LLVMContext llvmCtx;
  auto buffer = llvm::MemoryBuffer::getMemBufferCopy(
      sourceMgr.getMemoryBuffer(sourceMgr.getMainFileID())->getBuffer());
  std::unique_ptr<llvm::MIRParser> mirParser =
      llvm::createMIRParser(std::move(buffer), llvmCtx);
  if (!mirParser)
    return {};

  std::unique_ptr<llvm::Module> module = mirParser->parseIRModule();
  if (!module) {
    emitError(UnknownLoc::get(ctx)) << "failed to parse .mir IR module";
    return {};
  }

  // Construct an AArch64 TargetMachine (defaulting the triple if absent).
  llvm::Triple triple = module->getTargetTriple();
  if (triple.getArch() == llvm::Triple::UnknownArch)
    triple = llvm::Triple("aarch64-unknown-unknown");
  std::string err;
  const llvm::Target *target = llvm::TargetRegistry::lookupTarget(triple, err);
  if (!target) {
    emitError(UnknownLoc::get(ctx)) << "no target for triple " << triple.str()
                                    << ": " << err;
    return {};
  }
  llvm::TargetOptions options;
  std::unique_ptr<llvm::TargetMachine> tm(target->createTargetMachine(
      triple, "generic", "", options, /*RM=*/std::nullopt));
  module->setDataLayout(tm->createDataLayout());

  llvm::MachineModuleInfo mmi(tm.get());
  if (mirParser->parseMachineFunctions(*module, mmi)) {
    emitError(UnknownLoc::get(ctx)) << "failed to parse machine functions";
    return {};
  }

  OpBuilder b(ctx);
  ctx->getOrLoadDialect<mir::MIRDialect>();
  Location loc = UnknownLoc::get(ctx);
  OwningOpRef<ModuleOp> mlirModule = ModuleOp::create(loc);
  b.setInsertionPointToEnd(mlirModule->getBody());

  for (llvm::Function &f : *module) {
    llvm::MachineFunction *mf = mmi.getMachineFunction(f);
    if (!mf)
      continue;
    auto func = b.create<mir::FuncOp>(loc, f.getName());
    Block *body = &func.getBody().emplaceBlock();
    if (failed(Importer(ctx, *mf).run(body))) {
      emitError(loc) << "unsupported construct while importing @"
                     << f.getName();
      return {};
    }
  }

  return mlirModule.release().getOperation();
}

namespace mlir {
void registerFromMIRTranslation() {
  TranslateToMLIRRegistration registration(
      "import-mir", "Translate pre-RA LLVM MachineIR (.mir) to the MIR dialects",
      [](llvm::SourceMgr &sourceMgr,
         MLIRContext *ctx) -> OwningOpRef<Operation *> {
        return translateMIRToModule(sourceMgr, ctx);
      },
      [](DialectRegistry &registry) {
        registry.insert<mir::MIRDialect>();
      });
}
} // namespace mlir
