#ifndef LLDB_API_SBRPC_CHECKCONSTSBREF_H
#define LLDB_API_SBRPC_CHECKCONSTSBREF_H

namespace lldb {
class LLDB_API SBRPC_CheckConstSBRef {
public:
  // Const references to SB classes should be encoded as usual without
  // needing to create a new object with its own connection.
  int CheckConstSBRef(const SBDebugger &debugger_ref);

}; // class SBRPC_CheckConstSBRef
} // namespace lldb

#endif // LLDB_API_SBRPC_CHECKCONSTSBREF_H
