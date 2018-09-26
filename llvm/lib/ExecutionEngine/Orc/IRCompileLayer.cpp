//===--------------- IRCompileLayer.cpp - IR Compiling Layer --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"

namespace llvm {
namespace orc {

IRCompileLayer2::IRCompileLayer2(ExecutionSession &ES, ObjectLayer &BaseLayer,
                                 CompileFunction Compile)
    : IRLayer(ES), BaseLayer(BaseLayer), Compile(std::move(Compile)) {}

void IRCompileLayer2::setNotifyCompiled(NotifyCompiledFunction NotifyCompiled) {
  std::lock_guard<std::mutex> Lock(IRLayerMutex);
  this->NotifyCompiled = std::move(NotifyCompiled);
}

void IRCompileLayer2::emit(MaterializationResponsibility R, VModuleKey K,
                           ThreadSafeModule TSM) {
  assert(TSM.getModule() && "Module must not be null");

  if (auto Obj = Compile(*TSM.getModule())) {
    {
      std::lock_guard<std::mutex> Lock(IRLayerMutex);
      if (NotifyCompiled)
        NotifyCompiled(K, std::move(TSM));
      else
        TSM = ThreadSafeModule();
    }
    BaseLayer.emit(std::move(R), std::move(K), std::move(*Obj));
  } else {
    R.failMaterialization();
    getExecutionSession().reportError(Obj.takeError());
  }
}

} // End namespace orc.
} // End namespace llvm.
