//===- llvm/unittest/Bitcode/BitReaderTest.cpp - Tests for BitReader ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallString.h"
#include "llvm/Analysis/Verifier.h"
#include "llvm/Bitcode/BitstreamWriter.h"
#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/Constants.h"
#include "llvm/Instructions.h"
#include "llvm/LLVMContext.h"
#include "llvm/Module.h"
#include "llvm/PassManager.h"
#include "llvm/Support/MemoryBuffer.h"
#include "gtest/gtest.h"

namespace llvm {
namespace {

static Module *makeLLVMModule() {
  Module* Mod = new Module("test-mem", getGlobalContext());

  FunctionType* FuncTy =
    FunctionType::get(Type::getVoidTy(Mod->getContext()), false);
  Function* Func = Function::Create(FuncTy,GlobalValue::ExternalLinkage,
                                    "func", Mod);

  BasicBlock* Entry = BasicBlock::Create(Mod->getContext(), "entry", Func);
  new UnreachableInst(Mod->getContext(), Entry);

  BasicBlock* BB = BasicBlock::Create(Mod->getContext(), "bb", Func);
  new UnreachableInst(Mod->getContext(), BB);

  PointerType* Int8Ptr = Type::getInt8PtrTy(Mod->getContext());
  new GlobalVariable(*Mod, Int8Ptr, /*isConstant=*/true,
                     GlobalValue::ExternalLinkage,
                     BlockAddress::get(BB), "table");

  return Mod;
}

static void writeModuleToBuffer(SmallVectorImpl<char> &Buffer) {
  Module *Mod = makeLLVMModule();
  raw_svector_ostream OS(Buffer);
  WriteBitcodeToFile(Mod, OS);
}

TEST(BitReaderTest, MaterializeFunctionsForBlockAddr) { // PR11677
  SmallString<1024> Mem;
  writeModuleToBuffer(Mem);
  MemoryBuffer *Buffer = MemoryBuffer::getMemBuffer(Mem.str(), "test", false);
  std::string errMsg;
  Module *m = getLazyBitcodeModule(Buffer, getGlobalContext(), &errMsg);
  PassManager passes;
  passes.add(createVerifierPass());
  passes.run(*m);
}

}
}
