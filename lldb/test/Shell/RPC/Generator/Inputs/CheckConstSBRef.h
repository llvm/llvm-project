#ifndef LLDB_API_SBRPC_CHECKCONSTSBREF_H
#define LLDB_API_SBRPC_CHECKCONSTSBREF_H

#include <cstddef>
#include <cstdio>

#include "lldb/API/SBDefines.h"

namespace lldb {
class LLDB_API SBRPC_CHECKCONSTSBREF {
public:
  // Const references to SB classes should be encoded as usual without
  // needing to create a new object with its own connection.
  int CheckConstSBRef(const SBDebugger &debugger_ref);

}; // class SBRPC_CHECKCONSTSBREF
} // namespace lldb

#endif // LLDB_API_SBRPC_CHECKCONSTSBREF_H
