//=- RPCChannel.inc - LLVM out-of-process JIT execution for Windows --=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Implementation of the Windows-specific parts of the RPCChannel class
// which executes JITed code in a separate process from where it was built.
//
//===----------------------------------------------------------------------===//

namespace llvm {

bool RPCChannel::createServer() { return false; }

bool RPCChannel::createClient() { return false; }

bool RPCChannel::WriteBytes(const void *Data, size_t Size) { return false; }

bool RPCChannel::ReadBytes(void *Data, size_t Size) { return false; }

void RPCChannel::Wait() {}

RPCChannel::~RPCChannel() {}

} // namespace llvm
