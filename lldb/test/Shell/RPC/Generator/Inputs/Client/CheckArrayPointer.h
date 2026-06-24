#ifndef LLDB_API_SBRPC_CHECKARRAYPTR_H
#define LLDB_API_SBRPC_CHECKARRAYPTR_H

namespace lldb {
class SBRPC_CheckArrayPtr {
public:
  // Pointers to arrays followed by length must use a
  // Bytes object constructed using that pointer and the sizeof()
  // the array object.
  int CheckArrayPtr(int *array, int array_len);

}; // class SBRPC_CheckArrayPtr
} // namespace lldb

#endif // LLDB_API_SBRPC_CHECKARRAYPTR_H
