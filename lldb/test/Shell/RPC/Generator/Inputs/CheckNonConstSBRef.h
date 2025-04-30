#ifndef LLDB_API_SBRPC_CHECKNONCONSTSBREF_H
#define LLDB_API_SBRPC_CHECKNONCONSTSBREF_H

#include <cstddef>
#include <cstdio>

#include "lldb/API/SBDefines.h"

namespace lldb {
class LLDB_API SBRPC_CHECKNONCONSTSBREF {
public:
  // Non-const references to SB classes will have new objects
  // of that class constructed with the connection as the first parameter
  // before being encoded if their existing connection is invalid.
  int CheckNonConstSBRef(SBDebugger &debugger_ref);

}; // class SBRPC_CHECKNONCONSTSBREF
} // namespace lldb

#endif // LLDB_API_SBRPC_CHECKNONCONSTSBREF_H
