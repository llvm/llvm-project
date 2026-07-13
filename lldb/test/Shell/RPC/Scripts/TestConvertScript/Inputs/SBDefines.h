// This is a truncated version of SBDefines.h used to test that the script
// convert-lldb-header-to-rpc-header.py works correctly. The script changes LLDB references in
// the original file to RPC references.

// The include guard should change from LLDB_LLDB to LLDB_RPC.
// LLDB_API_SBDEFINES_H -> LLDB_RPC_SBDEFINES_H
#ifndef LLDB_API_SBDEFINES_H
#define LLDB_API_SBDEFINES_H

// Includes of public main LLDB headers should change to their RPC equivalents:
// "lldb/lldb-defines.h" -> "lldb-rpc-defines.h"
// Also, the includes for lldb-forward.h and lldb-versioning.h should be removed.
#include "lldb/lldb-defines.h"
#include "lldb/lldb-enumerations.h"
#include "lldb/lldb-forward.h"
#include "lldb/lldb-types.h"
#include "lldb/lldb-versioning.h"

// The comment that closes the include guard must change in the same way
// the original guard did.
// #endif // LLDB_API_SBDEFINES_H -> #endif // LLDB_RPC_API_SBDEFINES_H
#endif // LLDB_API_SBDEFINES_H
