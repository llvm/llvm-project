//===-------- OrcCBindingsStack.cpp - Orc JIT stack for C bindings --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "OrcCBindingsStack.h"

#include "llvm/ExecutionEngine/Orc/OrcABISupport.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/DynamicLibrary.h"
#include <cstdio>
#include <system_error>

using namespace llvm;

std::unique_ptr<OrcCBindingsStack::CompileCallbackMgr>
OrcCBindingsStack::createCompileCallbackMgr(Triple T) {
  switch (T.getArch()) {
  default:
    return nullptr;

  case Triple::x86: {
    typedef orc::LocalJITCompileCallbackManager<orc::OrcI386> CCMgrT;
    return llvm::make_unique<CCMgrT>(0);
  };

  case Triple::x86_64: {
    if ( T.getOS() == Triple::OSType::Win32 ) {
      typedef orc::LocalJITCompileCallbackManager<orc::OrcX86_64_Win32> CCMgrT;
      return llvm::make_unique<CCMgrT>(0);
    } else {
      typedef orc::LocalJITCompileCallbackManager<orc::OrcX86_64_SysV> CCMgrT;
      return llvm::make_unique<CCMgrT>(0);
    }
  }
  }
}

OrcCBindingsStack::IndirectStubsManagerBuilder
OrcCBindingsStack::createIndirectStubsMgrBuilder(Triple T) {
  switch (T.getArch()) {
  default:
    return nullptr;

  case Triple::x86:
    return []() {
      return llvm::make_unique<orc::LocalIndirectStubsManager<orc::OrcI386>>();
    };

  case Triple::x86_64:
    if (T.getOS() == Triple::OSType::Win32) {
      return [](){
        return llvm::make_unique<
          orc::LocalIndirectStubsManager<orc::OrcX86_64_Win32>>();
      };
    } else {
      return [](){
        return llvm::make_unique<
          orc::LocalIndirectStubsManager<orc::OrcX86_64_SysV>>();
      };
    }
  }
}
