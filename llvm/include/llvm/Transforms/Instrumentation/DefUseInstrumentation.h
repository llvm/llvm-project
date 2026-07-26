#ifndef LLVM_TRANSFORMS_INSTRUMENTATION_DEFUSEINSTRUMENTATION_H
#define LLVM_TRANSFORMS_INSTRUMENTATION_DEFUSEINSTRUMENTATION_H

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Use.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Transforms/IPO/SampleProfileProbe.h"
#include <cstdint>

namespace llvm {

class Module;

struct DefUseInstrumentationPass
    : PassInfoMixin<DefUseInstrumentationPass> {
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &) {

    LLVMContext& Ctx = M.getContext();
    IRBuilder<> Builder(Ctx);
    DenseMap<Instruction*, uint64_t> InstIDs;           // Мапа, для того чтоб повторный вызов инструкции вспоминался и айдишник ёё брался
    SmallVector<Instruction *> Instructions;            // Чтоб модуль заново не обходить, а по вектору пробежаться

    FunctionType* HookType = FunctionType::get(Type::getVoidTy(Ctx), {Type::getInt64Ty(Ctx)}, false);
    FunctionCallee Hook =  M.getOrInsertFunction("__def_use_trace_enter", HookType);


    // первый обход заполняет мапу инструкция - ID
    uint64_t CallID = 0;

    for (Function &F : M) {
      if (F.isDeclaration()) {
        continue;
      }
      for (BasicBlock &BB : F) {
        for (Instruction &I : BB) {
          Instructions.push_back(&I);
          InstIDs[&I] = CallID;
          CallID++;
        }
      }
    }
    // второй обход создает зависимости, на основе мапы, использует ли функция результат уже другой функции

    for (Instruction *I : Instructions) {
      uint64_t UseID = InstIDs.lookup(I);

      for (Use &Operand : I->operands()) {
        Value *V = Operand.get();

        Instruction *Def = dyn_cast<Instruction>(V);

        if (!Def) {
          continue;
        }

        if (!InstIDs.contains(Def))
          continue;

        uint64_t DefID = InstIDs.lookup(Def);
        
        errs()  <<    "DEF " << DefID <<
                      "-> USE " << UseID << "\n";
      }
    }

    // третий обход чтоб вставить колбеки
    for (Instruction *I : Instructions) {
      uint64_t ID = InstIDs.lookup(I);

      Builder.SetInsertPoint(I);
      Builder.CreateCall(Hook, Builder.getInt64(ID));
    }
        
    return PreservedAnalyses::none();
    }
  };

} // namespace llvm

#endif // LLVM_TRANSFORMS_INSTRUMENTATION_DEFUSEINSTRUMENTATION_H