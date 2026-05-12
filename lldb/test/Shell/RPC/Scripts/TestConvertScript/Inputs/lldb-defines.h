// This is a truncated version of lldb-defines.h used to test that the script
// convert-lldb-header-to-rpc-header.py works correctly. The script changes LLDB references in
// the original file to RPC references.

// The include guard should change from LLDB_LLDB to LLDB_RPC.
// LLDB_LLDB_DEFINES_H -> LLDB_RPC_DEFINES_H
#ifndef LLDB_LLDB_DEFINES_H
#define LLDB_LLDB_DEFINES_H

// Includes of public main LLDB headers should change to their RPC equivalents:
// "lldb/lldb-types.h" -> "lldb-rpc-types.h"
#include "lldb/lldb-types.h"

// The LLDB version must change from LLDB to LLDB_RPC
// LLDB_VERSION -> LLDB_RPC_VERSION
#define LLDB_VERSION 21
#define LLDB_REVISION 12
#define LLDB_VERSION_STRING "21.0.12"

// The comment that closes the include guard must change in the same way
// the original guard did.
// #endif // LLDB_LLDB_DEFINES_H -> #endif // LLDB_RPC_DEFINES_H
#endif // LLDB_LLDB_DEFINES_H
