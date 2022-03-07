//===- JMCInstrumenter.cpp - JMC Instrumentation --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// JMCInstrumenter pass:
// - add "/alternatename:__CheckForDebuggerJustMyCode=__JustMyCode_Default" to
//   "llvm.linker.options"
// - create the dummy COMDAT function __JustMyCode_Default
// - instrument each function with a call to __CheckForDebuggerJustMyCode. The
//   sole argument should be defined in .msvcjmc. Each flag is 1 byte initilized
//   to 1.
// - (TODO) currently targeting MSVC, adds ELF debuggers support
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/DJB.h"
#include "llvm/Support/Path.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

using namespace llvm;

#define DEBUG_TYPE "jmc-instrument"

namespace {
struct JMCInstrumenter : public ModulePass {
  static char ID;
  JMCInstrumenter() : ModulePass(ID) {
    initializeJMCInstrumenterPass(*PassRegistry::getPassRegistry());
  }
  bool runOnModule(Module &M) override;
};
char JMCInstrumenter::ID = 0;
} // namespace

INITIALIZE_PASS(
    JMCInstrumenter, DEBUG_TYPE,
    "Instrument function entry with call to __CheckForDebuggerJustMyCode",
    false, false)

ModulePass *llvm::createJMCInstrumenterPass() { return new JMCInstrumenter(); }

namespace {
const char CheckFunctionName[] = "__CheckForDebuggerJustMyCode";

std::string getFlagName(DISubprogram &SP, bool UseX86FastCall) {
  // Best effort path normalization. This is to guarantee an unique flag symbol
  // is produced for the same directory. Some builds may want to use relative
  // paths, or paths with a specific prefix (see the -fdebug-compilation-dir
  // flag), so only hash paths in debuginfo. Don't expand them to absolute
  // paths.
  SmallString<256> FilePath(SP.getDirectory());
  sys::path::append(FilePath, SP.getFilename());
  sys::path::native(FilePath);
  sys::path::remove_dots(FilePath, /*remove_dot_dot=*/true);

  // The naming convention for the flag name is __<hash>_<file name> with '.' in
  // <file name> replaced with '@'. For example C:\file.any.c would have a flag
  // __D032E919_file@any@c. The naming convention match MSVC's format however
  // the match is not required to make JMC work. The hashing function used here
  // is different from MSVC's.

  std::string Suffix;
  for (auto C : sys::path::filename(FilePath))
    Suffix.push_back(C == '.' ? '@' : C);

  sys::path::remove_filename(FilePath);
  return (UseX86FastCall ? "_" : "__") +
         utohexstr(djbHash(FilePath), /*LowerCase=*/false,
                   /*Width=*/8) +
         "_" + Suffix;
}

void attachDebugInfo(GlobalVariable &GV, DISubprogram &SP) {
  Module &M = *GV.getParent();
  DICompileUnit *CU = SP.getUnit();
  assert(CU);
  DIBuilder DB(M, false, CU);

  auto *DType =
      DB.createBasicType("unsigned char", 8, dwarf::DW_ATE_unsigned_char,
                         llvm::DINode::FlagArtificial);

  auto *DGVE = DB.createGlobalVariableExpression(
      CU, GV.getName(), /*LinkageName=*/StringRef(), SP.getFile(),
      /*LineNo=*/0, DType, /*IsLocalToUnit=*/true, /*IsDefined=*/true);
  GV.addMetadata(LLVMContext::MD_dbg, *DGVE);
  DB.finalize();
}

FunctionType *getCheckFunctionType(LLVMContext &Ctx) {
  Type *VoidTy = Type::getVoidTy(Ctx);
  PointerType *VoidPtrTy = Type::getInt8PtrTy(Ctx);
  return FunctionType::get(VoidTy, VoidPtrTy, false);
}

void createDefaultCheckFunction(Module &M, bool UseX86FastCall) {
  LLVMContext &Ctx = M.getContext();
  const char *DefaultCheckFunctionName =
      UseX86FastCall ? "_JustMyCode_Default" : "__JustMyCode_Default";
  // Create the function.
  Function *DefaultCheckFunc =
      Function::Create(getCheckFunctionType(Ctx), GlobalValue::ExternalLinkage,
                       DefaultCheckFunctionName, &M);
  DefaultCheckFunc->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
  DefaultCheckFunc->addParamAttr(0, Attribute::NoUndef);
  if (UseX86FastCall)
    DefaultCheckFunc->addParamAttr(0, Attribute::InReg);
  appendToUsed(M, {DefaultCheckFunc});
  Comdat *C = M.getOrInsertComdat(DefaultCheckFunctionName);
  C->setSelectionKind(Comdat::Any);
  DefaultCheckFunc->setComdat(C);
  BasicBlock *EntryBB = BasicBlock::Create(Ctx, "", DefaultCheckFunc);
  ReturnInst::Create(Ctx, EntryBB);

  // Add a linker option /alternatename to set the default implementation for
  // the check function.
  // https://devblogs.microsoft.com/oldnewthing/20200731-00/?p=104024
  std::string AltOption = std::string("/alternatename:") + CheckFunctionName +
                          "=" + DefaultCheckFunctionName;
  llvm::Metadata *Ops[] = {llvm::MDString::get(Ctx, AltOption)};
  MDTuple *N = MDNode::get(Ctx, Ops);
  M.getOrInsertNamedMetadata("llvm.linker.options")->addOperand(N);
}
} // namespace

bool JMCInstrumenter::runOnModule(Module &M) {
  bool Changed = false;
  LLVMContext &Ctx = M.getContext();
  Triple ModuleTriple(M.getTargetTriple());
  bool UseX86FastCall =
      ModuleTriple.isOSWindows() && ModuleTriple.getArch() == Triple::x86;

  Function *CheckFunction = nullptr;
  DenseMap<DISubprogram *, Constant *> SavedFlags(8);
  for (auto &F : M) {
    if (F.isDeclaration())
      continue;
    auto *SP = F.getSubprogram();
    if (!SP)
      continue;

    Constant *&Flag = SavedFlags[SP];
    if (!Flag) {
      std::string FlagName = getFlagName(*SP, UseX86FastCall);
      IntegerType *FlagTy = Type::getInt8Ty(Ctx);
      Flag = M.getOrInsertGlobal(FlagName, FlagTy, [&] {
        // FIXME: Put the GV in comdat and have linkonce_odr linkage to save
        //        .msvcjmc section space? maybe not worth it.
        GlobalVariable *GV = new GlobalVariable(
            M, FlagTy, /*isConstant=*/false, GlobalValue::InternalLinkage,
            ConstantInt::get(FlagTy, 1), FlagName);
        GV->setSection(".msvcjmc");
        GV->setAlignment(Align(1));
        GV->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
        attachDebugInfo(*GV, *SP);
        return GV;
      });
    }

    if (!CheckFunction) {
      assert(!M.getFunction(CheckFunctionName) &&
             "JMC instrument more than once?");
      CheckFunction = cast<Function>(
          M.getOrInsertFunction(CheckFunctionName, getCheckFunctionType(Ctx))
              .getCallee());
      CheckFunction->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
      CheckFunction->addParamAttr(0, Attribute::NoUndef);
      if (UseX86FastCall) {
        CheckFunction->setCallingConv(CallingConv::X86_FastCall);
        CheckFunction->addParamAttr(0, Attribute::InReg);
      }
    }
    // FIXME: it would be nice to make CI scheduling boundary, although in
    //        practice it does not matter much.
    auto *CI = CallInst::Create(CheckFunction, {Flag}, "",
                                &*F.begin()->getFirstInsertionPt());
    CI->addParamAttr(0, Attribute::NoUndef);
    if (UseX86FastCall) {
      CI->setCallingConv(CallingConv::X86_FastCall);
      CI->addParamAttr(0, Attribute::InReg);
    }

    Changed = true;
  }
  if (!Changed)
    return false;

  createDefaultCheckFunction(M, UseX86FastCall);
  return true;
}
