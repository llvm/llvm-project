//===------- OrcRemoteTargetRPCAPI.cpp - ORC Remote API utilities ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/OrcRemoteTargetRPCAPI.h"

#define PROCNAME(X) \
  case X ## Id: \
  return #X

namespace llvm {
namespace orc {
namespace remote {

const char *OrcRemoteTargetRPCAPI::getJITProcIdName(JITProcId Id) {
  switch (Id) {
  case InvalidId:
    return "*** Invalid JITProcId ***";
  PROCNAME(CallIntVoid);
  PROCNAME(CallIntVoidResponse);
  PROCNAME(CallMain);
  PROCNAME(CallMainResponse);
  PROCNAME(CallVoidVoid);
  PROCNAME(CallVoidVoidResponse);
  PROCNAME(CreateRemoteAllocator);
  PROCNAME(CreateIndirectStubsOwner);
  PROCNAME(DeregisterEHFrames);
  PROCNAME(DestroyRemoteAllocator);
  PROCNAME(DestroyIndirectStubsOwner);
  PROCNAME(EmitIndirectStubs);
  PROCNAME(EmitIndirectStubsResponse);
  PROCNAME(EmitResolverBlock);
  PROCNAME(EmitTrampolineBlock);
  PROCNAME(EmitTrampolineBlockResponse);
  PROCNAME(GetSymbolAddress);
  PROCNAME(GetSymbolAddressResponse);
  PROCNAME(GetRemoteInfo);
  PROCNAME(GetRemoteInfoResponse);
  PROCNAME(ReadMem);
  PROCNAME(ReadMemResponse);
  PROCNAME(RegisterEHFrames);
  PROCNAME(ReserveMem);
  PROCNAME(ReserveMemResponse);
  PROCNAME(RequestCompile);
  PROCNAME(RequestCompileResponse);
  PROCNAME(SetProtections);
  PROCNAME(TerminateSession);
  PROCNAME(WriteMem);
  PROCNAME(WritePtr);
  };
  return nullptr;
}
}
}
}
