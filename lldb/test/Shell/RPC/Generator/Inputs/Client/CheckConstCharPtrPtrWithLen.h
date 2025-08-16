#ifndef LLDB_API_SBRPC_CHECKCONSTCHARPTRPTRWITHLEN_H
#define LLDB_API_SBRPC_CHECKCONSTCHARPTRPTRWITHLEN_H

namespace lldb {
class SBRPC_CheckConstCharPtrPtrWithLen {
public:
  // const char ** followed by len must use a StringList
  // when being encoded.
  int CheckConstCharPtrPtrWithLen(const char **arg1, int len);

}; // class SBRPC_CheckConstCharPtrPtrWithLen
} // namespace lldb

#endif // LLDB_API_SBRPC_CHECKCONSTCHARPTRPTRWITHLEN_H
