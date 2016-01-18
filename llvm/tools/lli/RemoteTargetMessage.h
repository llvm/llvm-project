//===---- RemoteTargetMessage.h - LLI out-of-process message protocol -----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Definition of the LLIMessageType enum which is used for communication with a
// child process for remote execution.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLI_REMOTETARGETMESSAGE_H
#define LLVM_TOOLS_LLI_REMOTETARGETMESSAGE_H

namespace llvm {

// LLI messages from parent-to-child or vice versa follow an exceedingly simple
// protocol where the first four bytes represent the message type, the next
// four bytes represent the size of data for the command and following bytes
// represent the actual data.
//
// The protocol is not intended to be robust, secure or fault-tolerant.  It is
// only here for testing purposes and is therefore intended to be the simplest
// implementation that will work.  It is assumed that the parent and child
// share characteristics like endianness.
//
// Quick description of the protocol:
//
// { Header + Payload Size + Payload }
//
// The protocol message consist of a header, the payload size (which can be
// zero), and the payload itself. The payload can contain any number of items,
// and the size has to be the sum of them all. Each end is responsible for
// reading/writing the correct number of items with the correct sizes.
//
// The current four known exchanges are:
//
//  * Allocate Space:
//   Parent: { LLI_AllocateSpace, 8, Alignment, Size }
//    Child: { LLI_AllocationResult, 8, Address }
//
//  * Load Data:
//   Parent: { LLI_LoadDataSection, 8+Size, Address, Data }
//    Child: { LLI_LoadComplete, 4, StatusCode }
//
//  * Load Code:
//   Parent: { LLI_LoadCodeSection, 8+Size, Address, Code }
//    Child: { LLI_LoadComplete, 4, StatusCode }
//
//  * Execute Code:
//   Parent: { LLI_Execute, 8, Address }
//    Child: { LLI_ExecutionResult, 4, Result }
//
// It is the responsibility of either side to check for correct headers,
// sizes and payloads, since any inconsistency would misalign the pipe, and
// result in data corruption.

enum LLIMessageType {
  LLI_Error = -1,
  LLI_ChildActive = 0,        // Data = not used
  LLI_AllocateSpace,          // Data = struct { uint32_t Align, uint_32t Size }
  LLI_AllocationResult,       // Data = uint64_t Address (child memory space)

  LLI_LoadCodeSection,        // Data = uint64_t Address, void * SectionData
  LLI_LoadDataSection,        // Data = uint64_t Address, void * SectionData
  LLI_LoadResult,             // Data = uint32_t LLIMessageStatus

  LLI_Execute,                // Data = uint64_t Address
  LLI_ExecutionResult,        // Data = uint32_t Result

  LLI_Terminate               // Data = not used
};

enum LLIMessageStatus {
  LLI_Status_Success = 0,     // Operation succeeded
  LLI_Status_NotAllocated,    // Address+Size not allocated in child space
  LLI_Status_IncompleteMsg    // Size received doesn't match request
};

} // end namespace llvm

#endif
