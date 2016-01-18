//===---- RemoteTargetExternal.cpp - LLVM out-of-process JIT execution ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Implementation of the RemoteTargetExternal class which executes JITed code
// in a separate process from where it was built.
//
//===----------------------------------------------------------------------===//

#include "llvm/Config/config.h"
#include "RemoteTarget.h"
#include "RemoteTargetExternal.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/Memory.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"
#include <string>

using namespace llvm;

#define DEBUG_TYPE "lli"

bool RemoteTargetExternal::allocateSpace(size_t Size, unsigned Alignment,
                                 uint64_t &Address) {
  DEBUG(dbgs() << "Message [allocate space] size: " << Size <<
                  ", align: " << Alignment << "\n");
  if (!SendAllocateSpace(Alignment, Size)) {
    ErrorMsg += ", (RemoteTargetExternal::allocateSpace)";
    return false;
  }
  if (!Receive(LLI_AllocationResult, Address)) {
    ErrorMsg += ", (RemoteTargetExternal::allocateSpace)";
    return false;
  }
  if (Address == 0) {
    ErrorMsg += "failed allocation, (RemoteTargetExternal::allocateSpace)";
    return false;
  }
  DEBUG(dbgs() << "Message [allocate space] addr: 0x" <<
                  format("%llx", Address) << "\n");
  return true;
}

bool RemoteTargetExternal::loadData(uint64_t Address, const void *Data, size_t Size) {
  DEBUG(dbgs() << "Message [load data] addr: 0x" << format("%llx", Address) <<
                  ", size: " << Size << "\n");
  if (!SendLoadSection(Address, Data, (uint32_t)Size, false)) {
    ErrorMsg += ", (RemoteTargetExternal::loadData)";
    return false;
  }
  int Status = LLI_Status_Success;
  if (!Receive(LLI_LoadResult, Status)) {
    ErrorMsg += ", (RemoteTargetExternal::loadData)";
    return false;
  }
  if (Status == LLI_Status_IncompleteMsg) {
    ErrorMsg += "incomplete load data, (RemoteTargetExternal::loadData)";
    return false;
  }
  if (Status == LLI_Status_NotAllocated) {
    ErrorMsg += "data memory not allocated, (RemoteTargetExternal::loadData)";
    return false;
  }
  DEBUG(dbgs() << "Message [load data] complete\n");
  return true;
}

bool RemoteTargetExternal::loadCode(uint64_t Address, const void *Data, size_t Size) {
  DEBUG(dbgs() << "Message [load code] addr: 0x" << format("%llx", Address) <<
                  ", size: " << Size << "\n");
  if (!SendLoadSection(Address, Data, (uint32_t)Size, true)) {
    ErrorMsg += ", (RemoteTargetExternal::loadCode)";
    return false;
  }
  int Status = LLI_Status_Success;
  if (!Receive(LLI_LoadResult, Status)) {
    ErrorMsg += ", (RemoteTargetExternal::loadCode)";
    return false;
  }
  if (Status == LLI_Status_IncompleteMsg) {
    ErrorMsg += "incomplete load data, (RemoteTargetExternal::loadData)";
    return false;
  }
  if (Status == LLI_Status_NotAllocated) {
    ErrorMsg += "data memory not allocated, (RemoteTargetExternal::loadData)";
    return false;
  }
  DEBUG(dbgs() << "Message [load code] complete\n");
  return true;
}

bool RemoteTargetExternal::executeCode(uint64_t Address, int32_t &RetVal) {
  DEBUG(dbgs() << "Message [exectue code] addr: " << Address << "\n");
  if (!SendExecute(Address)) {
    ErrorMsg += ", (RemoteTargetExternal::executeCode)";
    return false;
  }
  if (!Receive(LLI_ExecutionResult, RetVal)) {
    ErrorMsg += ", (RemoteTargetExternal::executeCode)";
    return false;
  }
  DEBUG(dbgs() << "Message [exectue code] return: " << RetVal << "\n");
  return true;
}

void RemoteTargetExternal::stop() {
  SendTerminate();
  RPC.Wait();
}

bool RemoteTargetExternal::SendAllocateSpace(uint32_t Alignment, uint32_t Size) {
  if (!SendHeader(LLI_AllocateSpace)) {
    ErrorMsg += ", (RemoteTargetExternal::SendAllocateSpace)";
    return false;
  }

  AppendWrite((const void *)&Alignment, 4);
  AppendWrite((const void *)&Size, 4);

  if (!SendPayload()) {
    ErrorMsg += ", (RemoteTargetExternal::SendAllocateSpace)";
    return false;
  }
  return true;
}

bool RemoteTargetExternal::SendLoadSection(uint64_t Addr,
                                       const void *Data,
                                       uint32_t Size,
                                       bool IsCode) {
  LLIMessageType MsgType = IsCode ? LLI_LoadCodeSection : LLI_LoadDataSection;
  if (!SendHeader(MsgType)) {
    ErrorMsg += ", (RemoteTargetExternal::SendLoadSection)";
    return false;
  }

  AppendWrite((const void *)&Addr, 8);
  AppendWrite(Data, Size);

  if (!SendPayload()) {
    ErrorMsg += ", (RemoteTargetExternal::SendLoadSection)";
    return false;
  }
  return true;
}

bool RemoteTargetExternal::SendExecute(uint64_t Addr) {
  if (!SendHeader(LLI_Execute)) {
    ErrorMsg += ", (RemoteTargetExternal::SendExecute)";
    return false;
  }

  AppendWrite((const void *)&Addr, 8);

  if (!SendPayload()) {
    ErrorMsg += ", (RemoteTargetExternal::SendExecute)";
    return false;
  }
  return true;
}

bool RemoteTargetExternal::SendTerminate() {
  return SendHeader(LLI_Terminate);
  // No data or data size is sent with Terminate
}

bool RemoteTargetExternal::Receive(LLIMessageType Msg) {
  if (!ReceiveHeader(Msg))
    return false;
  int Unused;
  AppendRead(&Unused, 0);
  if (!ReceivePayload())
    return false;
  ReceiveData.clear();
  Sizes.clear();
  return true;
}

bool RemoteTargetExternal::Receive(LLIMessageType Msg, int32_t &Data) {
  if (!ReceiveHeader(Msg))
    return false;
  AppendRead(&Data, 4);
  if (!ReceivePayload())
    return false;
  ReceiveData.clear();
  Sizes.clear();
  return true;
}

bool RemoteTargetExternal::Receive(LLIMessageType Msg, uint64_t &Data) {
  if (!ReceiveHeader(Msg))
    return false;
  AppendRead(&Data, 8);
  if (!ReceivePayload())
    return false;
  ReceiveData.clear();
  Sizes.clear();
  return true;
}

bool RemoteTargetExternal::ReceiveHeader(LLIMessageType ExpectedMsgType) {
  assert(ReceiveData.empty() && Sizes.empty() &&
         "Payload vector not empty to receive header");

  // Message header, with type to follow
  uint32_t MsgType;
  if (!ReadBytes(&MsgType, 4)) {
    ErrorMsg += ", (RemoteTargetExternal::ReceiveHeader)";
    return false;
  }
  if (MsgType != (uint32_t)ExpectedMsgType) {
    ErrorMsg = "received unexpected message type";
    ErrorMsg += ". Expecting: ";
    ErrorMsg += ExpectedMsgType;
    ErrorMsg += ", Got: ";
    ErrorMsg += MsgType;
    return false;
  }
  return true;
}

bool RemoteTargetExternal::ReceivePayload() {
  assert(!ReceiveData.empty() &&
         "Payload vector empty to receive");
  assert(ReceiveData.size() == Sizes.size() &&
         "Unexpected mismatch between data and size");

  uint32_t TotalSize = 0;
  for (int I=0, E=Sizes.size(); I < E; I++)
    TotalSize += Sizes[I];

  // Payload size header
  uint32_t DataSize;
  if (!ReadBytes(&DataSize, 4)) {
    ErrorMsg += ", invalid data size";
    return false;
  }
  if (DataSize != TotalSize) {
    ErrorMsg = "unexpected data size";
    ErrorMsg += ". Expecting: ";
    ErrorMsg += TotalSize;
    ErrorMsg += ", Got: ";
    ErrorMsg += DataSize;
    return false;
  }
  if (DataSize == 0)
    return true;

  // Payload itself
  for (int I=0, E=Sizes.size(); I < E; I++) {
    if (!ReadBytes(ReceiveData[I], Sizes[I])) {
      ErrorMsg = "unexpected data while reading message";
      return false;
    }
  }

  return true;
}

bool RemoteTargetExternal::SendHeader(LLIMessageType MsgType) {
  assert(SendData.empty() && Sizes.empty() &&
         "Payload vector not empty to send header");

  // Message header, with type to follow
  if (!WriteBytes(&MsgType, 4)) {
    ErrorMsg += ", (RemoteTargetExternal::SendHeader)";
    return false;
  }
  return true;
}

bool RemoteTargetExternal::SendPayload() {
  assert(!SendData.empty() && !Sizes.empty() &&
         "Payload vector empty to send");
  assert(SendData.size() == Sizes.size() &&
         "Unexpected mismatch between data and size");

  uint32_t TotalSize = 0;
  for (int I=0, E=Sizes.size(); I < E; I++)
    TotalSize += Sizes[I];

  // Payload size header
  if (!WriteBytes(&TotalSize, 4)) {
    ErrorMsg += ", invalid data size";
    return false;
  }
  if (TotalSize == 0)
    return true;

  // Payload itself
  for (int I=0, E=Sizes.size(); I < E; I++) {
    if (!WriteBytes(SendData[I], Sizes[I])) {
      ErrorMsg = "unexpected data while writing message";
      return false;
    }
  }

  SendData.clear();
  Sizes.clear();
  return true;
}

void RemoteTargetExternal::AppendWrite(const void *Data, uint32_t Size) {
  SendData.push_back(Data);
  Sizes.push_back(Size);
}

void RemoteTargetExternal::AppendRead(void *Data, uint32_t Size) {
  ReceiveData.push_back(Data);
  Sizes.push_back(Size);
}

#ifdef LLVM_ON_UNIX
#include "Unix/RPCChannel.inc"
#endif

#ifdef LLVM_ON_WIN32
#include "Windows/RPCChannel.inc"
#endif
