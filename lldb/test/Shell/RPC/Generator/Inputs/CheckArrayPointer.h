#ifndef LLDB_API_SBRPC_CHECKARRAYPTR_H
#define LLDB_API_SBRPC_CHECKARRAYPTR_H

#include <cstddef>
#include <cstdio>

#include "lldb/API/SBDefines.h"

namespace lldb {
class LLDB_API SBRPC_CHECKARRAYPTR {
public:
  // Pointers to arrays followed by length must use a
  // Bytes object constructed using that pointer and the sizeof()
  // the array object.
  int CheckArrayPtr(uint64_t *array, size_t array_len);

}; // class SBRPC_CHECKARRAYPTR
} // namespace lldb

#endif // LLDB_API_SBRPC_CHECKARRAYPTR_H
