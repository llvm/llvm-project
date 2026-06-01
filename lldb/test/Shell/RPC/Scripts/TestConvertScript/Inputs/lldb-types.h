// This is a truncated version of lldb-types.h used to test that the script
// convert-lldb-header-to-rpc-header.py works correctly. The script changes LLDB references in
// the original file to RPC references.

// The include guard should change from LLDB_LLDB to LLDB_RPC.
// LLDB_LLDB_TYPES_H -> LLDB_RPC_TYPES_H
#ifndef LLDB_LLDB_TYPES_H
#define LLDB_LLDB_TYPES_H

// Includes of public main LLDB headers should change to their RPC equivalents:
// "lldb/lldb-defines.h" -> "lldb-rpc-defines.h":
// Also, the includes for lldb-forward.h should be removed.
#include "lldb/lldb-enumerations.h"
#include "lldb/lldb-forward.h"

// The namespace definition should change to the lldb_rpc namespace, so should the comment that closes it:
// namespace lldb -> namespace lldb_rpc
namespace lldb {} // namespace lldb

// The comment that closes the include guard must change in the same way
// the original guard did:
// #endif // LLDB_LLDB_TYPES_H -> #endif // LLDB_RPC_TYPES_H
#endif // LLDB_LLDB_TYPES_H
