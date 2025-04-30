#ifndef LLDB_API_SBRPC_CHECKCONSTCHARPTRPTRWITHLEN_H
#define LLDB_API_SBRPC_CHECKCONSTCHARPTRPTRWITHLEN_H

#include <cstddef>
#include <cstdio>

#include "lldb/API/SBDefines.h"

namespace lldb {
class LLDB_API SBRPC_CHECKCONSTCHARPTRPTRWITHLEN {
public:
  // const char ** followed by len must use a StringList
  // when being encoded.
  int CheckConstCharPtrPtrWithLen(const char **arg1, size_t len);

}; // class SBRPC_CHECKCONSTCHARPTRPTRWITHLEN
} // namespace lldb

#endif // LLDB_API_SBRPC_CHECKCONSTCHARPTRPTRWITHLEN_H
