#ifndef LLDB_API_SBRPC_CHECKVOIDPTR_H
#define LLDB_API_SBRPC_CHECKVOIDPTR_H

#include "lldb/API/SBDefines.h"

namespace lldb {
class LLDB_API SBRPC_CheckVoidPtr {
public:
  // void * followed by length must use a Bytes object
  // when being encoded.
  int CheckVoidPtr(void *buf, int len);

}; // class SBRPC_CheckVoidPtr
} // namespace lldb

#endif // LLDB_API_SBRPC_CHECKVOIDPTR_H
