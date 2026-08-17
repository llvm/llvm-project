//===-- llvm/Target/TargetOptions.h - Target Options ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines command line option flags that are shared across various
// targets.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_TARGETOPTIONS_H
#define LLVM_TARGET_TARGETOPTIONS_H

#include "llvm/ADT/FloatingPointMode.h"
#include "llvm/IR/SystemLibraries.h"
#include "llvm/MC/MCTargetOptions.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/Compiler.h"

#include <memory>
#include <string>

namespace llvm {
struct fltSemantics;
class MachineFunction;
class MemoryBuffer;

namespace FPOpFusion {
enum FPOpFusionMode {
  Fast,     // Enable fusion of FP ops wherever it's profitable.
  Standard, // Only allow fusion of 'blessed' ops (currently just fmuladd).
  Strict    // Never fuse FP-ops.
};
}

namespace JumpTable {
enum JumpTableType {
  Single,     // Use a single table for all indirect jumptable calls.
  Arity,      // Use one table per number of function parameters.
  Simplified, // Use one table per function type, with types projected
              // into 4 types: pointer to non-function, struct,
              // primitive, and function pointer.
  Full        // Use one table per unique function type
};
}

namespace ThreadModel {
enum Model {
  POSIX, // POSIX Threads
  Single // Single Threaded Environment
};
}

enum class BasicBlockSection {
  All,    // Use Basic Block Sections for all basic blocks.  A section
          // for every basic block can significantly bloat object file sizes.
  List,   // Get list of functions & BBs from a file. Selectively enables
          // basic block sections for a subset of basic blocks which can be
          // used to control object size bloats from creating sections.
  Preset, // Similar to list but the blocks are identified by passes which
          // seek to use Basic Block Sections, e.g. MachineFunctionSplitter.
          // This option cannot be set via the command line.
  None    // Do not use Basic Block Sections.
};

/// Identify a debugger for "tuning" the debug info.
///
/// The "debugger tuning" concept allows us to present a more intuitive
/// interface that unpacks into different sets of defaults for the various
/// individual feature-flag settings, that suit the preferences of the
/// various debuggers.  However, it's worth remembering that debuggers are
/// not the only consumers of debug info, and some variations in DWARF might
/// better be treated as target/platform issues. Fundamentally,
/// o if the feature is useful (or not) to a particular debugger, regardless
///   of the target, that's a tuning decision;
/// o if the feature is useful (or not) on a particular platform, regardless
///   of the debugger, that's a target decision.
/// It's not impossible to see both factors in some specific case.
enum class DebuggerKind {
  Default, ///< No specific tuning requested.
  GDB,     ///< Tune debug info for gdb.
  LLDB,    ///< Tune debug info for lldb.
  SCE,     ///< Tune debug info for SCE targets (e.g. PS4).
  DBX      ///< Tune debug info for dbx.
};

/// Enable abort calls when global instruction selection fails to lower/select
/// an instruction.
enum class GlobalISelAbortMode {
  Disable,        // Disable the abort.
  Enable,         // Enable the abort.
  DisableWithDiag // Disable the abort but emit a diagnostic on failure.
};

/// Indicates when and how the Swift async frame pointer bit should be set.
enum class SwiftAsyncFramePointerMode {
  /// Determine whether to set the bit statically or dynamically based
  /// on the deployment target.
  DeploymentBased,
  /// Always set the bit.
  Always,
  /// Never set the bit.
  Never,
};

/// \brief Enumeration value for AMDGPU code object version, which is the
/// code object version times 100.
enum CodeObjectVersionKind {
  COV_None,
  COV_2 = 200, // Unsupported.
  COV_3 = 300, // Unsupported.
  COV_4 = 400,
  COV_5 = 500,
  COV_6 = 600,
};

class TargetOptions {
public:
  TargetOptions() {
#define TARGET_OPTION_INIT_BOOL(Type, Name, Bits, Default) Name = Default;
#define TARGET_OPTION_INIT_U32_BITFIELD(Type, Name, Bits, Default)             \
  Name = Default;
#define TARGET_OPTION_INIT_PAIR(Type, Name, Bits, Default)
#define TARGET_OPTION_INIT_ENUM(Type, Name, Bits, Default)
#define TARGET_OPTION_INIT_STRING(Type, Name, Bits, Default)
#define TARGET_OPTION_INIT_U32(Type, Name, Bits, Default)
#define TARGET_OPTION_INIT_BUFFER(Type, Name, Bits, Default)
#define TARGET_OPTION_INIT_MC(Type, Name, Bits, Default)
#define TARGET_OPTION(Type, Name, Bits, Default, Kind)                         \
  TARGET_OPTION_INIT_##Kind(Type, Name, Bits, Default)
#include "llvm/Target/TargetOptions.def"
#undef TARGET_OPTION_INIT_BOOL
#undef TARGET_OPTION_INIT_U32_BITFIELD
#undef TARGET_OPTION_INIT_PAIR
#undef TARGET_OPTION_INIT_ENUM
#undef TARGET_OPTION_INIT_STRING
#undef TARGET_OPTION_INIT_U32
#undef TARGET_OPTION_INIT_BUFFER
#undef TARGET_OPTION_INIT_MC
  }

  LLVM_ABI bool HonorSignDependentRoundingFPMath() const;

  /// NOTE: There are targets that still do not support the debug entry values
  /// production.
  LLVM_ABI bool ShouldEmitDebugEntryValues() const;

#define TARGET_OPTION_TYPE(...) __VA_ARGS__
#define TARGET_OPTION_DECLARE_BOOL(Type, Name, Bits, Default)                  \
  TARGET_OPTION_TYPE Type Name : Bits;
#define TARGET_OPTION_DECLARE_U32_BITFIELD(Type, Name, Bits, Default)          \
  TARGET_OPTION_TYPE Type Name : Bits;
#define TARGET_OPTION_DECLARE_PAIR(Type, Name, Bits, Default)                  \
  TARGET_OPTION_TYPE Type Name = Default;
#define TARGET_OPTION_DECLARE_ENUM(Type, Name, Bits, Default)                  \
  TARGET_OPTION_TYPE Type Name = Default;
#define TARGET_OPTION_DECLARE_STRING(Type, Name, Bits, Default)                \
  TARGET_OPTION_TYPE Type Name = Default;
#define TARGET_OPTION_DECLARE_U32(Type, Name, Bits, Default)                   \
  TARGET_OPTION_TYPE Type Name = Default;
#define TARGET_OPTION_DECLARE_BUFFER(Type, Name, Bits, Default)                \
  TARGET_OPTION_TYPE Type Name = Default;
#define TARGET_OPTION_DECLARE_MC(Type, Name, Bits, Default)                    \
  TARGET_OPTION_TYPE Type Name = Default;
#define TARGET_OPTION(Type, Name, Bits, Default, Kind)                         \
  TARGET_OPTION_DECLARE_##Kind(Type, Name, Bits, Default)
#include "llvm/Target/TargetOptions.def"
#undef TARGET_OPTION_TYPE
#undef TARGET_OPTION_DECLARE_BOOL
#undef TARGET_OPTION_DECLARE_U32_BITFIELD
#undef TARGET_OPTION_DECLARE_PAIR
#undef TARGET_OPTION_DECLARE_ENUM
#undef TARGET_OPTION_DECLARE_STRING
#undef TARGET_OPTION_DECLARE_U32
#undef TARGET_OPTION_DECLARE_BUFFER
#undef TARGET_OPTION_DECLARE_MC
};

} // namespace llvm

#endif
