//===-- MSILWriter.h - TargetMachine for the MSIL ---------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the MSILWriter that is used by the MSIL.
//
//===----------------------------------------------------------------------===//
#ifndef MSILWRITER_H
#define MSILWRITER_H

#include "llvm/Constants.h"
#include "llvm/Module.h"
#include "llvm/Instructions.h"
#include "llvm/IntrinsicInst.h"
#include "llvm/Pass.h"
#include "llvm/PassManager.h"
#include "llvm/Analysis/FindUsedTypes.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Support/GetElementPtrTypeIterator.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetMachineRegistry.h"
#include "llvm/Support/Mangler.h"
#include <ios>
using namespace llvm;

namespace {

  class MSILModule : public ModulePass {
    Module *ModulePtr;
    const std::set<const Type *>*& UsedTypes;
    const TargetData*& TD;

  public:
    static char ID;
    MSILModule(const std::set<const Type *>*& _UsedTypes,
               const TargetData*& _TD)
      : ModulePass(&ID), UsedTypes(_UsedTypes), TD(_TD) {}

    void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequired<FindUsedTypes>();
      AU.addRequired<TargetData>();
    }

    virtual const char *getPassName() const {
      return "MSIL backend definitions";
    }

    virtual bool runOnModule(Module &M);

  };

  class MSILWriter  : public FunctionPass {
    struct StaticInitializer {
      const Constant* constant;
      uint64_t offset;
      
      StaticInitializer()
        : constant(0), offset(0) {}

      StaticInitializer(const Constant* _constant, uint64_t _offset)
        : constant(_constant), offset(_offset) {} 
    };

    uint64_t UniqID;

    uint64_t getUniqID() {
      return ++UniqID;
    }

  public:
    raw_ostream &Out;
    Module* ModulePtr;
    const TargetData* TD;
    Mangler* Mang;
    LoopInfo *LInfo;
    std::vector<StaticInitializer>* InitListPtr;
    std::map<const GlobalVariable*,std::vector<StaticInitializer> >
      StaticInitList;
    const std::set<const Type *>* UsedTypes;
    static char ID;
    MSILWriter(raw_ostream &o) : FunctionPass(&ID), Out(o) {
      UniqID = 0;
    }

    enum ValueType {
      UndefVT,
      GlobalVT,
      InternalVT,
      ArgumentVT,
      LocalVT,
      ConstVT,
      ConstExprVT
    };

    bool isVariable(ValueType V) {
      return V==GlobalVT || V==InternalVT || V==ArgumentVT || V==LocalVT;
    }

    bool isConstValue(ValueType V) {
      return V==ConstVT || V==ConstExprVT;
    }

    virtual const char *getPassName() const { return "MSIL backend"; }

    void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequired<LoopInfo>();
      AU.setPreservesAll();
    }

    bool runOnFunction(Function &F);

    virtual bool doInitialization(Module &M);

    virtual bool doFinalization(Module &M);

    void printModuleStartup();

    bool isZeroValue(const Value* V);

    std::string getValueName(const Value* V);

    std::string getLabelName(const Value* V);

    std::string getLabelName(const std::string& Name);

    std::string getConvModopt(unsigned CallingConvID);

    std::string getArrayTypeName(Type::TypeID TyID, const Type* Ty);

    std::string getPrimitiveTypeName(const Type* Ty, bool isSigned);

    std::string getFunctionTypeName(const Type* Ty);

    std::string getPointerTypeName(const Type* Ty);

    std::string getTypeName(const Type* Ty, bool isSigned = false,
                            bool isNested = false);

    ValueType getValueLocation(const Value* V);

    std::string getTypePostfix(const Type* Ty, bool Expand,
                               bool isSigned = false);

    void printConvToPtr();

    void printPtrLoad(uint64_t N);

    void printValuePtrLoad(const Value* V);

    void printConstLoad(const Constant* C);

    void printValueLoad(const Value* V);

    void printValueSave(const Value* V);

    void printBinaryInstruction(const char* Name, const Value* Left,
                                const Value* Right);

    void printSimpleInstruction(const char* Inst, const char* Operand = NULL);

    void printPHICopy(const BasicBlock* Src, const BasicBlock* Dst);

    void printBranchToBlock(const BasicBlock* CurrBB,
                            const BasicBlock* TrueBB,
                            const BasicBlock* FalseBB);

    void printBranchInstruction(const BranchInst* Inst);

    void printSelectInstruction(const Value* Cond, const Value* VTrue,
                                const Value* VFalse);

    void printIndirectLoad(const Value* V);

    void printIndirectSave(const Value* Ptr, const Value* Val);

    void printIndirectSave(const Type* Ty);

    void printCastInstruction(unsigned int Op, const Value* V,
                              const Type* Ty);

    void printGepInstruction(const Value* V, gep_type_iterator I,
                             gep_type_iterator E);

    std::string getCallSignature(const FunctionType* Ty,
                                 const Instruction* Inst,
                                 std::string Name);

    void printFunctionCall(const Value* FnVal, const Instruction* Inst);

    void printIntrinsicCall(const IntrinsicInst* Inst);

    void printCallInstruction(const Instruction* Inst);

    void printICmpInstruction(unsigned Predicate, const Value* Left,
                              const Value* Right);

    void printFCmpInstruction(unsigned Predicate, const Value* Left,
                              const Value* Right);

    void printInvokeInstruction(const InvokeInst* Inst);

    void printSwitchInstruction(const SwitchInst* Inst);

    void printVAArgInstruction(const VAArgInst* Inst);

    void printAllocaInstruction(const AllocaInst* Inst);

    void printInstruction(const Instruction* Inst);

    void printLoop(const Loop* L);

    void printBasicBlock(const BasicBlock* BB);
    
    void printLocalVariables(const Function& F);

    void printFunctionBody(const Function& F);

    void printConstantExpr(const ConstantExpr* CE);

    void printStaticInitializerList();

    void printFunction(const Function& F);

    void printDeclarations(const TypeSymbolTable& ST);

    unsigned int getBitWidth(const Type* Ty);

    void printStaticConstant(const Constant* C, uint64_t& Offset);

    void printStaticInitializer(const Constant* C, const std::string& Name);

    void printVariableDefinition(const GlobalVariable* G);

    void printGlobalVariables();

    const char* getLibraryName(const Function* F);

    const char* getLibraryName(const GlobalVariable* GV); 
    
    const char* getLibraryForSymbol(const char* Name, bool isFunction,
                                    unsigned CallingConv);

    void printExternals();
  };
}

#endif

