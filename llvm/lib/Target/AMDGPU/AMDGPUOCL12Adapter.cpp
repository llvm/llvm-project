//==- AMDGPUOCL12Adapter.cpp - Fix OpenCL1.2 builtin calls for user Module -*- C++ -*-===//
//
// Copyright(c) 2014 Advanced Micro Devices, Inc. All rights reserved.
//
//===----------------------------------------------------------------------===//
//
/// \file
/// \brief Provide pass to convert OpenCL 1.2 builtin function calls in user kernel
///  to its corresponding 2.0 function call.
//
///  1.2 Builtin function calls in user kernel are mangled and need to be changed
///  to the corresponding 2.0 mangled name. Pointer arguments in 1.2 calls are
///  address space specific, and are translated to the generic address space for
///  2.0 calls.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "AMDGPUOCL12Adapter"

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "AMDGPU.h"
#include <string>

using namespace llvm;

namespace llvm {
class AMDGPUOCL12Adapter : public ModulePass {

public:
  static char ID;

  AMDGPUOCL12Adapter() : ModulePass(ID) {
    initializeAMDGPUOCL12AdapterPass(*PassRegistry::getPassRegistry());
  }

  virtual bool runOnModule(Module &M);
  };
}

INITIALIZE_PASS(AMDGPUOCL12Adapter, "amdgpu-opencl-12-adapter",
                "Convert OpenCL 1.2 builtins to 2.0 builtins", false, false)

char AMDGPUOCL12Adapter::ID = 0;

namespace llvm {
ModulePass *createAMDGPUOCL12AdapterPass() { return new AMDGPUOCL12Adapter(); }
}

char &llvm::AMDGPUOCL12AdapterID = AMDGPUOCL12Adapter::ID;

/// \brief Check whether the type is a pointer and also whether it points to
/// non-default address space. If it is not an opaque type, return true.
/// Always skip opaque types because they are not "real" pointers.
static bool isNonDefaultAddrSpacePtr(Type *Ty, AMDGPUAS AMDGPUASI) {
  PointerType *PtrType = dyn_cast<PointerType>(Ty);
  if(!PtrType)
    return false;
  StructType* StrType = dyn_cast<StructType>(PtrType->getElementType());
  if(StrType && StrType->isOpaque())
    return false;
  return (PtrType->getAddressSpace() != AMDGPUASI.FLAT_ADDRESS &&
          PtrType->getAddressSpace() != AMDGPUASI.CONSTANT_ADDRESS);
}

/// \brief Check whether the Function signature has any of the
/// non-default address space pointers as arguments. If yes,
/// this funtion will return true.
static bool hasNonDefaultAddrSpaceArg(const Function *F, AMDGPUAS AMDGPUASI) {

  for (const Argument &AI: F->args())
    if (isNonDefaultAddrSpacePtr(AI.getType(), AMDGPUASI))
      return true;
  return false;
}

/// \brief Locate the position of the function name in the mangled OpenCL
/// builtin function. Returns true on failure.
static bool locateFuncName(StringRef FuncName, size_t &FuncNameStart,
                           size_t &FuncNameSize) {

  // Find the first non-digit number in the mangled name of the
  // builtin.
  // The search should start from "2" because first two characters
  // are "_Z" in the mangling scheme.
  size_t NumStartPos = 2;
  FuncNameStart = FuncName.find_first_not_of("0123456789", NumStartPos);
  // Extract the integer, which is equal to the number of chars
  // in the function name.
  StringRef SizeInChar = FuncName.slice(NumStartPos, FuncNameStart);
  return SizeInChar.getAsInteger(/*radix=*/10, FuncNameSize);
}

/// \brief Returns the declaration of the builtin function
///  with all the address space of the arguments are "4".
///  Name mangling is also modified accordingly to match the
///  defintion in the OpenCL2.0 builtins library.
static Function *getNewOCL20BuiltinFuncDecl(Function *OldFunc,
    AMDGPUAS AMDGPUASI) {

  bool GIZ = AMDGPUASI.FLAT_ADDRESS == 0;
  size_t FuncNameStart, FuncNameSize;
  std::string MangledFuncName = OldFunc->getName();
  locateFuncName(OldFunc->getName(),FuncNameStart,FuncNameSize);

  std::string FuncName = MangledFuncName.substr(FuncNameStart,FuncNameSize);
  std::string NewFuncName =  MangledFuncName;

  size_t StartIndexPos = FuncNameStart + FuncNameSize;
  size_t tmp = StartIndexPos;
  for (; StartIndexPos < NewFuncName.size(); StartIndexPos++) {
    // Find the Address space pointer arguments in the mangled name.
    // Replace all address pointers with generic address space
    StartIndexPos = NewFuncName.find("P", StartIndexPos);
    if (StartIndexPos == std::string::npos)
      break;
    else {
      // Skip in cases where CV qualifiers are used: r, V, K
      tmp = NewFuncName.find("U3AS", StartIndexPos);
      bool HasNonZeroAddr = tmp != std::string::npos && tmp <= StartIndexPos+3;
      if (GIZ) {
        if (HasNonZeroAddr) {
          NewFuncName.erase(tmp, 5);
        }
      } else {
        char GenAddr = '0' + AMDGPUASI.FLAT_ADDRESS;
        if (HasNonZeroAddr)
            NewFuncName.at(tmp+4) = GenAddr;
        else {
          NewFuncName.insert(StartIndexPos + 1, "U3AS");
          NewFuncName.insert(StartIndexPos + 5, 1, GenAddr);
        }

        StartIndexPos += 5;
      }
    }
  }

  // Create the arguments vector for new Function.
  SmallVector<Type *, 1> NewFuncArgs;
  for (Function::arg_iterator AI = OldFunc->arg_begin(), E = OldFunc->arg_end();
    AI!= E; ++AI) {
    Type *ArgType = AI->getType();

    if (!isNonDefaultAddrSpacePtr(ArgType, AMDGPUASI)) {
      NewFuncArgs.push_back(ArgType);
      continue;
    }

    PointerType *PtrType = cast<PointerType>(ArgType);
    Type *EleType = PtrType->getElementType();
    PointerType *NewPtrType = PointerType::get(EleType, AMDGPUASI.FLAT_ADDRESS);
    NewFuncArgs.push_back(NewPtrType);
  }

  FunctionType *NewFuncType = FunctionType::get(
      OldFunc->getReturnType(), NewFuncArgs, OldFunc->isVarArg());
  Module *M = OldFunc->getParent();
  Value *NewFunc = M->getOrInsertFunction(NewFuncName, NewFuncType);
  if (Function *Fn = dyn_cast<Function>(NewFunc->stripPointerCasts())) {
    Fn->setCallingConv(OldFunc->getCallingConv());
    return Fn;
  }
  return NULL;
}

/// \brief Define the 1.2 OpenCL builtin called by the user to call the
/// OpenCL 2.0 builtin which has only generic address space arguments.
void createOCL20BuiltinFuncDefn(Function *OldFunc, Function *NewFunc,
    AMDGPUAS AMDGPUASI) {

  // Adding alwaysinline attribute for the adapter function.
  OldFunc->addFnAttr(Attribute::AlwaysInline);
  BasicBlock *EntryBlock =
      BasicBlock::Create(OldFunc->getContext(), "entry", OldFunc);
  IRBuilder<> BBBuilder(EntryBlock);
  SmallVector<llvm::Value *, 1> NewFuncCallArgs;

  for (auto &Arg : OldFunc->args()) {
    if (!isNonDefaultAddrSpacePtr(Arg.getType(), AMDGPUASI)) {
      NewFuncCallArgs.push_back(&Arg);
      continue;
    }

    PointerType *PtrType = cast<PointerType>(Arg.getType());
    Type *EleType = PtrType->getElementType();
    PointerType *NewPtrType = PointerType::get(EleType, AMDGPUASI.FLAT_ADDRESS);

    // Cast all non-default addr space pointer arguments to default addr
    // space pointers. Note that this cast will result in no-op.
    Value *CastVal = BBBuilder.
      CreatePointerBitCastOrAddrSpaceCast(&Arg, NewPtrType);
    NewFuncCallArgs.push_back(CastVal);
  }
  Value *CallInstVal = BBBuilder.CreateCall(NewFunc, NewFuncCallArgs);
  if (CallInstVal->getType()->isVoidTy()) {
    BBBuilder.CreateRetVoid();
    return;
  }
  BBBuilder.CreateRet(CallInstVal);
  OldFunc->setLinkage(GlobalValue::LinkOnceODRLinkage);
  return;
}

/// \brief Generate right function calls for all "undefined" 1.2 OpenCL builtins
/// in the whole Module. Returns true if at least one of the 1.2 OpenCL builtin
/// has been modified.
static bool findAndDefineBuiltinCalls(Module &M) {
  auto AMDGPUASI = AMDGPU::getAMDGPUAS(M);
  bool isModified = false;
  for (auto &F : M) {

    // Search only for used, undefined OpenCL builtin functions,
    // which has non-default addr space pointer arguments.
    if (!F.empty() || F.use_empty() || !F.getName().startswith("_Z") ||
        !hasNonDefaultAddrSpaceArg(&F, AMDGPUASI))
      continue;
    if (F.getName().find("async_work_group", 0) == StringRef::npos &&
        F.getName().find("prefetch", 0) == StringRef::npos) {
      isModified = true;
      Function *NewFunc = getNewOCL20BuiltinFuncDecl(&F, AMDGPUASI);
      // Get the new Function declaration.
      DEBUG(dbgs() << "\n Modifying Func " << F.getName() << " to call "
       << NewFunc->getName() << " Function");
      createOCL20BuiltinFuncDefn(&F, NewFunc, AMDGPUASI);
    }
  }
  return isModified;
}

bool AMDGPUOCL12Adapter::runOnModule(Module &M) {
  // Do not translate modules from languages other than OpenCL.
  const char *const OCLVersionMDName = "opencl.ocl.version";
  if (!M.getNamedMetadata(OCLVersionMDName))
    return false;
  return findAndDefineBuiltinCalls(M);
}
