#ifndef LLDB_API_SBRPC_CHECKCONSTCHARPOINTER_H
#define LLDB_API_SBRPC_CHECKCONSTCHARPOINTER_H

namespace lldb {
class LLDB_API SBRPC_CHECKCONSTCHARPOINTER {
public:
  // const char * parameters must decoded as rpc_common::ConstCharPointer in server side
  // source files.
  int CheckConstCharPointer(char *string);

}; // class SBRPC_CHECKCONSTCHARPOINTER
} // namespace lldb

#endif // LLDB_API_SBRPC_CHECKCONSTCHARPOINTER_H
