//===-------- OrcCBindingsStack.cpp - Orc JIT stack for C bindings --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "OrcCBindingsStack.h"

#include "llvm/ExecutionEngine/Orc/OrcTargetSupport.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/DynamicLibrary.h"
#include <cstdio>
#include <system_error>

using namespace llvm;

std::unique_ptr<OrcCBindingsStack::CompileCallbackMgr>
OrcCBindingsStack::createCompileCallbackMgr(Triple T) {
  switch (T.getArch()) {
    default: return nullptr;

    case Triple::x86_64: {
      typedef orc::LocalJITCompileCallbackManager<orc::OrcX86_64> CCMgrT;
      return llvm::make_unique<CCMgrT>(0);
    }
  }
}

OrcCBindingsStack::IndirectStubsManagerBuilder
OrcCBindingsStack::createIndirectStubsMgrBuilder(Triple T) {
  switch (T.getArch()) {
    default: return nullptr;

    case Triple::x86_64:
      return [](){
        return llvm::make_unique<
                 orc::LocalIndirectStubsManager<orc::OrcX86_64>>();
      };
  }
}
