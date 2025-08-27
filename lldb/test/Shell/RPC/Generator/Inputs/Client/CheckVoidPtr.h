#ifndef LLDB_API_SBRPC_CHECKVOIDPTR_H
#define LLDB_API_SBRPC_CHECKVOIDPTR_H

namespace lldb {
class SBRPC_CheckVoidPtr {
public:
  // void * followed by length must use a Bytes object
  // when being encoded.
  int CheckVoidPtr(void *buf, int len);

}; // class SBRPC_CheckVoidPtr
} // namespace lldb

#endif // LLDB_API_SBRPC_CHECKVOIDPTR_H
