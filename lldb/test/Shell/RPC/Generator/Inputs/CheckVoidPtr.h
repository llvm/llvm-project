#ifndef LLDB_API_SBRPC_CHECKVOIDPTR_H
#define LLDB_API_SBRPC_CHECKVOIDPTR_H

#include <cstddef>
#include <cstdio>

#include "lldb/API/SBDefines.h"

namespace lldb {
class LLDB_API SBRPC_CHECKVOIDPTR {
public:
  // void * followed by length must use a Bytes object
  // when being encoded.
  int CheckVoidPtr(void *buf, size_t len);

}; // class SBRPC_CHECKVOIDPTR
} // namespace lldb

#endif // LLDB_API_SBRPC_CHECKVOIDPTR_H
