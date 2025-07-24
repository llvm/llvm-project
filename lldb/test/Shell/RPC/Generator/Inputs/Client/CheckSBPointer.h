#ifndef LLDB_API_SBRPC_CHECKSBPTR_H
#define LLDB_API_SBRPC_CHECKSBPTR_H

#include "lldb/API/SBDefines.h"

namespace lldb {
class LLDB_API SBRPC_CheckSBPtr {
public:
  // Pointers to SB objects must be checked to
  // see if they're null. If so, then a new object of the given
  // class must be created and encoded. Otherwise, the original
  // parameter will be encoded.
  int CheckSBPtr(SBDebugger *debugger_ptr);

}; // class SBRPC_CheckSBPtr
} // namespace lldb

#endif // LLDB_API_SBRPC_CHECKSBPTR_H
