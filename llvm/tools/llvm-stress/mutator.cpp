#include "mutator.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/CodeGen/CommandFlags.h"
#include "llvm/FuzzMutate/FuzzerCLI.h"
#include "llvm/FuzzMutate/IRMutator.h"
#include "llvm/FuzzMutate/Operations.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"

#include <fstream>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#ifdef DEBUG
#include "llvm/IR/Verifier.h"
#include "llvm/Transforms/Utils/Cloning.h"
#endif
using namespace llvm;

static std::unique_ptr<IRMutator> Mutator;

extern "C" {

void dumpOnFailure(unsigned int Seed, uint8_t *Data, size_t Size,
                   size_t MaxSize) {
  time_t seconds = time(NULL);
  errs() << "Mutation failed, seed: " << Seed << "\n";
  char oldname[256];
  memset(oldname, 0, 256);
  sprintf(oldname, "%u-%zu-%zu.old.bc", Seed, MaxSize, seconds);
  std::ofstream oldoutfile =
      std::ofstream(oldname, std::ios::out | std::ios::binary);
  oldoutfile.write((char *)Data, Size);
  oldoutfile.close();
}

void addVectorTypeGetters(std::vector<TypeGetter> &Types) {
  int VectorLength[] = {1, 2, 4, 8, 16, 32};
  std::vector<TypeGetter> BasicTypeGetters(Types);
  for (auto typeGetter : BasicTypeGetters) {
    for (int length : VectorLength) {
      Types.push_back([typeGetter, length](LLVMContext &C) {
        return VectorType::get(typeGetter(C), length, false);
      });
    }
  }
}

/// TODO:
/// Type* getStructType(Context& C);

void createISelMutator() {
  std::vector<TypeGetter> Types{
      Type::getInt1Ty,  Type::getInt8Ty,  Type::getInt16Ty, Type::getInt32Ty,
      Type::getInt64Ty, Type::getFloatTy, Type::getDoubleTy};
  std::vector<TypeGetter> ScalarTypes = Types;

  addVectorTypeGetters(Types);

  TypeGetter OpaquePtrGetter = [](LLVMContext &C) {
    return PointerType::get(Type::getInt32Ty(C), 0);
  };
  Types.push_back(OpaquePtrGetter);

  // Copy scalar types to change distribution.
  for (int i = 0; i < 5; i++)
    Types.insert(Types.end(), ScalarTypes.begin(), ScalarTypes.end());

  std::vector<std::unique_ptr<IRMutationStrategy>> Strategies;
  std::vector<fuzzerop::OpDescriptor> Ops = InjectorIRStrategy::getDefaultOps();

  Strategies.push_back(std::make_unique<InjectorIRStrategy>(
      InjectorIRStrategy::getDefaultOps()));
  Strategies.push_back(std::make_unique<InstDeleterIRStrategy>());
  Strategies.push_back(std::make_unique<InstModificationIRStrategy>());
  // Strategies.push_back(std::make_unique<FunctionIRStrategy>());
  Strategies.push_back(std::make_unique<InsertCFGStrategy>());
  Strategies.push_back(std::make_unique<InsertPHIStrategy>());
  Strategies.push_back(std::make_unique<SinkInstructionStrategy>());
  Strategies.push_back(std::make_unique<ShuffleBlockStrategy>());

  Mutator =
      std::make_unique<IRMutator>(std::move(Types), std::move(Strategies));
}

size_t LLVMFuzzerCustomMutator(uint8_t *Data, size_t Size, size_t MaxSize,
                               unsigned int Seed) {
  LLVMContext Context;
  std::unique_ptr<Module> M;
  if (Size <= 1)
    // We get bogus data given an empty corpus - just create a new module.
    M.reset(new Module("M", Context));
  else
    M = parseModule(Data, Size, Context);
  if (!M) {
    errs() << "Parse module error. No mutation is done. Data size: " << Size
           << ". Given data wrote to err.bc\n";
    std::ofstream outfile =
        std::ofstream("err.bc", std::ios::out | std::ios::binary);
    outfile.write((char *)Data, Size);
    outfile.close();
#ifdef DEBUG
    exit(1);
#else
    // We don't do any change.
    return Size;
#endif
  }
#ifdef DEBUG
  std::unique_ptr<Module> OldM = CloneModule(*M);
#endif

#ifdef DEBUG
  try {
#endif
    srand(Seed);
    Seed = rand();
    // for (int i = 0; i < 4; i++) {
    Mutator->mutateModule(*M, Seed, Size, MaxSize);
    // }
#ifdef DEBUG
  } catch (...) {
    dumpOnFailure(Seed, Data, Size, MaxSize);
    return Size;
  }
#endif

#ifdef DEBUG
  uint8_t NewData[MaxSize];
  size_t NewSize = writeModule(*M, NewData, MaxSize);
  LLVMContext NewC;
  auto NewM = parseModule(NewData, NewSize, NewC);
  if (NewM == nullptr) {
    dumpOnFailure(Seed, Data, Size, MaxSize);
    return Size;
  } else {
    memset(Data, 0, MaxSize);
    memcpy(Data, NewData, NewSize);
    return NewSize;
  }
#else
  return writeModule(*M, Data, MaxSize);
#endif
}
}