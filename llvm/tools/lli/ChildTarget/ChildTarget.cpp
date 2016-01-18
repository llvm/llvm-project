#include "llvm/Config/config.h"
#include "../RPCChannel.h"
#include "../RemoteTarget.h"
#include "../RemoteTargetMessage.h"
#include "llvm/Support/Memory.h"
#include <assert.h>
#include <map>
#include <stdint.h>
#include <string>
#include <vector>

using namespace llvm;

class LLIChildTarget {
public:
  void initialize();
  LLIMessageType waitForIncomingMessage();
  void handleMessage(LLIMessageType messageType);
  RemoteTarget *RT;
  RPCChannel RPC;

private:
  // Incoming message handlers
  void handleAllocateSpace();
  void handleLoadSection(bool IsCode);
  void handleExecute();

  // Outgoing message handlers
  void sendChildActive();
  void sendAllocationResult(uint64_t Addr);
  void sendLoadStatus(uint32_t Status);
  void sendExecutionComplete(int Result);

  // OS-specific functions
  void initializeConnection();
  int WriteBytes(const void *Data, size_t Size) {
    return RPC.WriteBytes(Data, Size) ? Size : -1;
  }
  int ReadBytes(void *Data, size_t Size) {
    return RPC.ReadBytes(Data, Size) ? Size : -1;
  }

  // Communication handles (OS-specific)
  void *ConnectionData;
};

int main() {
  LLIChildTarget  ThisChild;
  ThisChild.RT = new RemoteTarget();
  ThisChild.initialize();
  LLIMessageType MsgType;
  do {
    MsgType = ThisChild.waitForIncomingMessage();
    ThisChild.handleMessage(MsgType);
  } while (MsgType != LLI_Terminate &&
           MsgType != LLI_Error);
  delete ThisChild.RT;
  return 0;
}

// Public methods
void LLIChildTarget::initialize() {
  RPC.createClient();
  sendChildActive();
}

LLIMessageType LLIChildTarget::waitForIncomingMessage() {
  int32_t MsgType = -1;
  if (ReadBytes(&MsgType, 4) > 0)
    return (LLIMessageType)MsgType;
  return LLI_Error;
}

void LLIChildTarget::handleMessage(LLIMessageType messageType) {
  switch (messageType) {
    case LLI_AllocateSpace:
      handleAllocateSpace();
      break;
    case LLI_LoadCodeSection:
      handleLoadSection(true);
      break;
    case LLI_LoadDataSection:
      handleLoadSection(false);
      break;
    case LLI_Execute:
      handleExecute();
      break;
    case LLI_Terminate:
      RT->stop();
      break;
    default:
      // FIXME: Handle error!
      break;
  }
}

// Incoming message handlers
void LLIChildTarget::handleAllocateSpace() {
  // Read and verify the message data size.
  uint32_t DataSize = 0;
  int rc = ReadBytes(&DataSize, 4);
  (void)rc;
  assert(rc == 4);
  assert(DataSize == 8);

  // Read the message arguments.
  uint32_t Alignment = 0;
  uint32_t AllocSize = 0;
  rc = ReadBytes(&Alignment, 4);
  assert(rc == 4);
  rc = ReadBytes(&AllocSize, 4);
  assert(rc == 4);

  // Allocate the memory.
  uint64_t Addr;
  RT->allocateSpace(AllocSize, Alignment, Addr);

  // Send AllocationResult message.
  sendAllocationResult(Addr);
}

void LLIChildTarget::handleLoadSection(bool IsCode) {
  // Read the message data size.
  uint32_t DataSize = 0;
  int rc = ReadBytes(&DataSize, 4);
  (void)rc;
  assert(rc == 4);

  // Read the target load address.
  uint64_t Addr = 0;
  rc = ReadBytes(&Addr, 8);
  assert(rc == 8);
  size_t BufferSize = DataSize - 8;

  if (!RT->isAllocatedMemory(Addr, BufferSize))
    return sendLoadStatus(LLI_Status_NotAllocated);

  // Read section data into previously allocated buffer
  rc = ReadBytes((void*)Addr, BufferSize);
  if (rc != (int)(BufferSize))
    return sendLoadStatus(LLI_Status_IncompleteMsg);

  // If IsCode, mark memory executable
  if (IsCode)
    sys::Memory::InvalidateInstructionCache((void *)Addr, BufferSize);

  // Send MarkLoadComplete message.
  sendLoadStatus(LLI_Status_Success);
}

void LLIChildTarget::handleExecute() {
  // Read the message data size.
  uint32_t DataSize = 0;
  int rc = ReadBytes(&DataSize, 4);
  (void)rc;
  assert(rc == 4);
  assert(DataSize == 8);

  // Read the target address.
  uint64_t Addr = 0;
  rc = ReadBytes(&Addr, 8);
  assert(rc == 8);

  // Call function
  int32_t Result = -1;
  RT->executeCode(Addr, Result);

  // Send ExecutionResult message.
  sendExecutionComplete(Result);
}

// Outgoing message handlers
void LLIChildTarget::sendChildActive() {
  // Write the message type.
  uint32_t MsgType = (uint32_t)LLI_ChildActive;
  int rc = WriteBytes(&MsgType, 4);
  (void)rc;
  assert(rc == 4);

  // Write the data size.
  uint32_t DataSize = 0;
  rc = WriteBytes(&DataSize, 4);
  assert(rc == 4);
}

void LLIChildTarget::sendAllocationResult(uint64_t Addr) {
  // Write the message type.
  uint32_t MsgType = (uint32_t)LLI_AllocationResult;
  int rc = WriteBytes(&MsgType, 4);
  (void)rc;
  assert(rc == 4);

  // Write the data size.
  uint32_t DataSize = 8;
  rc = WriteBytes(&DataSize, 4);
  assert(rc == 4);

  // Write the allocated address.
  rc = WriteBytes(&Addr, 8);
  assert(rc == 8);
}

void LLIChildTarget::sendLoadStatus(uint32_t Status) {
  // Write the message type.
  uint32_t MsgType = (uint32_t)LLI_LoadResult;
  int rc = WriteBytes(&MsgType, 4);
  (void)rc;
  assert(rc == 4);

  // Write the data size.
  uint32_t DataSize = 4;
  rc = WriteBytes(&DataSize, 4);
  assert(rc == 4);

  // Write the result.
  rc = WriteBytes(&Status, 4);
  assert(rc == 4);
}

void LLIChildTarget::sendExecutionComplete(int Result) {
  // Write the message type.
  uint32_t MsgType = (uint32_t)LLI_ExecutionResult;
  int rc = WriteBytes(&MsgType, 4);
  (void)rc;
  assert(rc == 4);


  // Write the data size.
  uint32_t DataSize = 4;
  rc = WriteBytes(&DataSize, 4);
  assert(rc == 4);

  // Write the result.
  rc = WriteBytes(&Result, 4);
  assert(rc == 4);
}

#ifdef LLVM_ON_UNIX
#include "../Unix/RPCChannel.inc"
#endif

#ifdef LLVM_ON_WIN32
#include "../Windows/RPCChannel.inc"
#endif
