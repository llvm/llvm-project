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

const char *OrcRemoteTargetRPCAPI::getJITProcIdName(JITProcId Id) {
  switch (Id) {
  case InvalidId:
    return "*** Invalid JITProcId ***";
  case CallIntVoidId:
    return "CallIntVoid";
  case CallIntVoidResponseId:
    return "CallIntVoidResponse";
  case CallMainId:
    return "CallMain";
  case CallMainResponseId:
    return "CallMainResponse";
  case CallVoidVoidId:
    return "CallVoidVoid";
  case CallVoidVoidResponseId:
    return "CallVoidVoidResponse";
  case CreateRemoteAllocatorId:
    return "CreateRemoteAllocator";
  case CreateIndirectStubsOwnerId:
    return "CreateIndirectStubsOwner";
  case DestroyRemoteAllocatorId:
    return "DestroyRemoteAllocator";
  case DestroyIndirectStubsOwnerId:
    return "DestroyIndirectStubsOwner";
  case EmitIndirectStubsId:
    return "EmitIndirectStubs";
  case EmitIndirectStubsResponseId:
    return "EmitIndirectStubsResponse";
  case EmitResolverBlockId:
    return "EmitResolverBlock";
  case EmitTrampolineBlockId:
    return "EmitTrampolineBlock";
  case EmitTrampolineBlockResponseId:
    return "EmitTrampolineBlockResponse";
  case GetSymbolAddressId:
    return "GetSymbolAddress";
  case GetSymbolAddressResponseId:
    return "GetSymbolAddressResponse";
  case GetRemoteInfoId:
    return "GetRemoteInfo";
  case GetRemoteInfoResponseId:
    return "GetRemoteInfoResponse";
  case ReadMemId:
    return "ReadMem";
  case ReadMemResponseId:
    return "ReadMemResponse";
  case ReserveMemId:
    return "ReserveMem";
  case ReserveMemResponseId:
    return "ReserveMemResponse";
  case RequestCompileId:
    return "RequestCompile";
  case RequestCompileResponseId:
    return "RequestCompileResponse";
  case SetProtectionsId:
    return "SetProtections";
  case TerminateSessionId:
    return "TerminateSession";
  case WriteMemId:
    return "WriteMem";
  case WritePtrId:
    return "WritePtr";
  };
  return nullptr;
}
}
}
}
