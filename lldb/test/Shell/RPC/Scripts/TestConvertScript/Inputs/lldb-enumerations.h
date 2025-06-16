// This is a truncated version of lldb-enumerations.h used to test that the script
// convert-lldb-header-to-rpc-header.py works correctly. The script changes LLDB references in
// the original file to RPC references.

// The include guard should change from LLDB_LLDB to LLDB_RPC.
// LLDB_LLDB_ENUMERATIONS_H -> LLDB_RPC_ENUMERATIONS_H
#ifndef LLDB_LLDB_ENUMERATIONS_H
#define LLDB_LLDB_ENUMERATIONS_H

// The namespace definition should change to the lldb_rpc namespace, so should the comment that closes it:
// namespace lldb -> namespace lldb_rpc
namespace lldb {} // namespace lldb

// The comment that closes the include guard must change in the same way
// the original guard did:
// #endif // LLDB_LLDB_ENUMERATIONS_H -> #endif // LLDB_RPC_ENUMERATIONS_H
#endif // LLDB_LLDB_ENUMERATIONS_H
