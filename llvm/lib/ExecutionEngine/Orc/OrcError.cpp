//===---------------- OrcError.cpp - Error codes for ORC ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Error codes for ORC.
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/OrcError.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ManagedStatic.h"

using namespace llvm;
using namespace llvm::orc;

namespace {

class OrcErrorCategory : public std::error_category {
public:
  const char *name() const LLVM_NOEXCEPT override { return "orc"; }

  std::string message(int condition) const override {
    switch (static_cast<OrcErrorCode>(condition)) {
    case OrcErrorCode::RemoteAllocatorDoesNotExist:
      return "Remote allocator does not exist";
    case OrcErrorCode::RemoteAllocatorIdAlreadyInUse:
      return "Remote allocator Id already in use";
    case OrcErrorCode::RemoteMProtectAddrUnrecognized:
      return "Remote mprotect call references unallocated memory";
    case OrcErrorCode::RemoteIndirectStubsOwnerDoesNotExist:
      return "Remote indirect stubs owner does not exist";
    case OrcErrorCode::RemoteIndirectStubsOwnerIdAlreadyInUse:
      return "Remote indirect stubs owner Id already in use";
    case OrcErrorCode::UnexpectedRPCCall:
      return "Unexpected RPC call";
    }
    llvm_unreachable("Unhandled error code");
  }
};

static ManagedStatic<OrcErrorCategory> OrcErrCat;
}

namespace llvm {
namespace orc {

std::error_code orcError(OrcErrorCode ErrCode) {
  typedef std::underlying_type<OrcErrorCode>::type UT;
  return std::error_code(static_cast<UT>(ErrCode), *OrcErrCat);
}
}
}
