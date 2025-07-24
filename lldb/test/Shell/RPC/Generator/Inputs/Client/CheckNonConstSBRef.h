#ifndef LLDB_API_SBRPC_CHECKNONCONSTSBREF_H
#define LLDB_API_SBRPC_CHECKNONCONSTSBREF_H

#include "lldb/API/SBDefines.h"

namespace lldb {
class LLDB_API SBRPC_CheckNonConstSBRef {
public:
  // Non-const references to SB classes will have an assert
  // added that trips if it has an invalid ObjectRef.
  int CheckNonConstSBRef(SBDebugger &debugger_ref);

}; // class SBRPC_CheckNonConstSBRef
} // namespace lldb

#endif // LLDB_API_SBRPC_CHECKNONCONSTSBREF_H
