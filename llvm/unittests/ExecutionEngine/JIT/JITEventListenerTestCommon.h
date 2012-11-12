//===- JITEventListenerTestCommon.h - Helper for JITEventListener tests ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===-------------------------------------------------------------------------------===//

#ifndef JIT_EVENT_LISTENER_TEST_COMMON_H
#define JIT_EVENT_LISTENER_TEST_COMMON_H

#include "llvm/DIBuilder.h"
#include "llvm/DebugInfo.h"
#include "llvm/IRBuilder.h"
#include "llvm/Instructions.h"
#include "llvm/Module.h"
#include "llvm/TypeBuilder.h"
#include "llvm/CodeGen/MachineCodeInfo.h"
#include "llvm/ExecutionEngine/JIT.h"
#include "llvm/ExecutionEngine/JITEventListener.h"
#include "llvm/Support/Dwarf.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Config/config.h"

#include "gtest/gtest.h"

#include <vector>
#include <string>
#include <utility>

typedef std::vector<std::pair<std::string, unsigned int> > SourceLocations;
typedef std::map<uint64_t, SourceLocations> NativeCodeMap;

class JITEnvironment : public testing::Environment {
  virtual void SetUp() {
    // Required to create a JIT.
    llvm::InitializeNativeTarget();
  }
};

inline unsigned int getLine() {
  return 12;
}

inline unsigned int getCol() {
  return 0;
}

inline const char* getFilename() {
  return "mock_source_file.cpp";
}

// Test fixture shared by tests for listener implementations
template<typename WrapperT>
class JITEventListenerTestBase : public testing::Test {
protected:
  llvm::OwningPtr<WrapperT> MockWrapper;
  llvm::OwningPtr<llvm::JITEventListener> Listener;

public:
  llvm::Module* M;
  llvm::MDNode* Scope;
  llvm::ExecutionEngine* EE;
  llvm::DIBuilder* DebugBuilder;
  llvm::IRBuilder<> Builder;

  JITEventListenerTestBase(WrapperT* w)
  : MockWrapper(w)
  , M(new llvm::Module("module", llvm::getGlobalContext()))
  , EE(llvm::EngineBuilder(M)
    .setEngineKind(llvm::EngineKind::JIT)
    .setOptLevel(llvm::CodeGenOpt::None)
    .create())
  , DebugBuilder(new llvm::DIBuilder(*M))
  , Builder(llvm::getGlobalContext())
  {
    DebugBuilder->createCompileUnit(llvm::dwarf::DW_LANG_C_plus_plus,
                                    "JIT",
                                    "JIT",
                                    "JIT",
                                    true,
                                    "",
                                    1);

    Scope = DebugBuilder->createFile(getFilename(), ".");
  }

  llvm::Function *buildFunction(const SourceLocations& DebugLocations) {
    using namespace llvm;

    LLVMContext& GlobalContext = getGlobalContext();

    SourceLocations::const_iterator CurrentDebugLocation
      = DebugLocations.begin();

    if (CurrentDebugLocation != DebugLocations.end()) {
      DebugLoc DebugLocation = DebugLoc::get(getLine(), getCol(),
          DebugBuilder->createFile(CurrentDebugLocation->first, "."));
      Builder.SetCurrentDebugLocation(DebugLocation);
      CurrentDebugLocation++;
    }

    Function *Result = Function::Create(
        TypeBuilder<int32_t(int32_t), false>::get(GlobalContext),
        GlobalValue::ExternalLinkage, "id", M);
    Value *Arg = Result->arg_begin();
    BasicBlock *BB = BasicBlock::Create(M->getContext(), "entry", Result);
    Builder.SetInsertPoint(BB);
    Value* one = ConstantInt::get(GlobalContext, APInt(32, 1));
    for(; CurrentDebugLocation != DebugLocations.end();
        ++CurrentDebugLocation) {
      Arg = Builder.CreateMul(Arg, Builder.CreateAdd(Arg, one));
      Builder.SetCurrentDebugLocation(
        DebugLoc::get(CurrentDebugLocation->second, 0,
                      DebugBuilder->createFile(CurrentDebugLocation->first, ".")));
    }
    Builder.CreateRet(Arg);
    return Result;
  }

  void TestNoDebugInfo(NativeCodeMap& ReportedDebugFuncs) {
    SourceLocations DebugLocations;
    llvm::Function* f = buildFunction(DebugLocations);
    EXPECT_TRUE(0 != f);

    //Cause JITting and callbacks to our listener
    EXPECT_TRUE(0 != EE->getPointerToFunction(f));
    EXPECT_TRUE(1 == ReportedDebugFuncs.size());

    EE->freeMachineCodeForFunction(f);
    EXPECT_TRUE(ReportedDebugFuncs.size() == 0);
  }

  void TestSingleLine(NativeCodeMap& ReportedDebugFuncs) {
    SourceLocations DebugLocations;
    DebugLocations.push_back(std::make_pair(std::string(getFilename()),
                                            getLine()));
    llvm::Function* f = buildFunction(DebugLocations);
    EXPECT_TRUE(0 != f);

    EXPECT_TRUE(0 != EE->getPointerToFunction(f));
    EXPECT_TRUE(1 == ReportedDebugFuncs.size());
    EXPECT_STREQ(ReportedDebugFuncs.begin()->second.begin()->first.c_str(),
                 getFilename());
    EXPECT_EQ(ReportedDebugFuncs.begin()->second.begin()->second, getLine());

    EE->freeMachineCodeForFunction(f);
    EXPECT_TRUE(ReportedDebugFuncs.size() == 0);
  }

  void TestMultipleLines(NativeCodeMap& ReportedDebugFuncs) {
    using namespace std;

    SourceLocations DebugLocations;
    unsigned int c = 5;
    for(unsigned int i = 0; i < c; ++i) {
      DebugLocations.push_back(make_pair(string(getFilename()), getLine() + i));
    }

    llvm::Function* f = buildFunction(DebugLocations);
    EXPECT_TRUE(0 != f);

    EXPECT_TRUE(0 != EE->getPointerToFunction(f));
    EXPECT_TRUE(1 == ReportedDebugFuncs.size());
    SourceLocations& FunctionInfo = ReportedDebugFuncs.begin()->second;
    EXPECT_EQ(c, FunctionInfo.size());

    int VerifyCount = 0;
    for(SourceLocations::iterator i = FunctionInfo.begin();
        i != FunctionInfo.end();
        ++i) {
      EXPECT_STREQ(i->first.c_str(), getFilename());
      EXPECT_EQ(i->second, getLine() + VerifyCount);
      VerifyCount++;
    }

    EE->freeMachineCodeForFunction(f);
    EXPECT_TRUE(ReportedDebugFuncs.size() == 0);
  }

  void TestMultipleFiles(NativeCodeMap& ReportedDebugFuncs) {

    std::string secondFilename("another_file.cpp");

    SourceLocations DebugLocations;
    DebugLocations.push_back(std::make_pair(std::string(getFilename()),
                                            getLine()));
    DebugLocations.push_back(std::make_pair(secondFilename, getLine()));
    llvm::Function* f = buildFunction(DebugLocations);
    EXPECT_TRUE(0 != f);

    EXPECT_TRUE(0 != EE->getPointerToFunction(f));
    EXPECT_TRUE(1 == ReportedDebugFuncs.size());
    SourceLocations& FunctionInfo = ReportedDebugFuncs.begin()->second;
    EXPECT_TRUE(2 == FunctionInfo.size());

    EXPECT_STREQ(FunctionInfo.at(0).first.c_str(), getFilename());
    EXPECT_STREQ(FunctionInfo.at(1).first.c_str(), secondFilename.c_str());

    EXPECT_EQ(FunctionInfo.at(0).second, getLine());
    EXPECT_EQ(FunctionInfo.at(1).second, getLine());

    EE->freeMachineCodeForFunction(f);
    EXPECT_TRUE(ReportedDebugFuncs.size() == 0);
  }
};

#endif //JIT_EVENT_LISTENER_TEST_COMMON_H
