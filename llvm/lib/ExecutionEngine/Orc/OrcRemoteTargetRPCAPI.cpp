//===------- OrcRemoteTargetRPCAPI.cpp - ORC Remote API utilities ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/OrcRemoteTargetRPCAPI.h"

namespace llvm {
namespace orc {
namespace remote {

#define FUNCNAME(X) \
  case X ## Id: \
  return #X

const char *OrcRemoteTargetRPCAPI::getJITFuncIdName(JITFuncId Id) {
  switch (Id) {
  case InvalidId:
    return "*** Invalid JITFuncId ***";
  FUNCNAME(CallIntVoid);
  FUNCNAME(CallMain);
  FUNCNAME(CallVoidVoid);
  FUNCNAME(CreateRemoteAllocator);
  FUNCNAME(CreateIndirectStubsOwner);
  FUNCNAME(DeregisterEHFrames);
  FUNCNAME(DestroyRemoteAllocator);
  FUNCNAME(DestroyIndirectStubsOwner);
  FUNCNAME(EmitIndirectStubs);
  FUNCNAME(EmitResolverBlock);
  FUNCNAME(EmitTrampolineBlock);
  FUNCNAME(GetSymbolAddress);
  FUNCNAME(GetRemoteInfo);
  FUNCNAME(ReadMem);
  FUNCNAME(RegisterEHFrames);
  FUNCNAME(ReserveMem);
  FUNCNAME(RequestCompile);
  FUNCNAME(SetProtections);
  FUNCNAME(TerminateSession);
  FUNCNAME(WriteMem);
  FUNCNAME(WritePtr);
  };
  return nullptr;
}

#undef FUNCNAME

} // end namespace remote
} // end namespace orc
} // end namespace llvm
