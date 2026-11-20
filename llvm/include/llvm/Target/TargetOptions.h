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

namespace llvm {
struct fltSemantics;
class MachineFunction;
class MemoryBuffer;

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

enum class BasicBlockSection {
  All,    // Use Basic Block Sections for all basic blocks.  A section
          // for every basic block can significantly bloat object file sizes.
  List,   // Get list of functions & BBs from a file. Selectively enables
          // basic block sections for a subset of basic blocks which can be
          // used to control object size bloats from creating sections.
  Preset, // Similar to list but the blocks are identified by passes which
          // seek to use Basic Block Sections, e.g. -function-splitting=all.
          // This option cannot be set via the command line.
  None    // Do not use Basic Block Sections.
};

/// Late function splitting mode. Decides which functions are eligible to have
/// their cold blocks moved into a separate section.
enum class FunctionSplittingMode {
  None,           // Hard off. Never create a cold section, even for functions
                  // which have a basic block sections profile. Such functions
                  // are still laid out using the profile, but are emitted as a
                  // single contiguous section.
  BBSectionsOnly, // Split only functions that have a basic block sections
                  // profile. Functions without one are not split, even if
                  // they have PGO/SamplePGO data. This is the default, and
                  // matches the behavior without -fsplit-machine-functions.
  All             // Split functions using the basic block sections profile
                  // where it is available, and PGO/AutoFDO elsewhere.
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
  TargetOptions()
      : NoZerosInBSS(false), GuaranteedTailCallOpt(false),
        StackSymbolOrdering(true), EnableFastISel(false),
        EnableGlobalISel(false), UseInitArray(false), FunctionSections(false),
        DataSections(false), IgnoreXCOFFVisibility(false),
        XCOFFTracebackTable(true), UniqueSectionNames(true),
        UniqueBasicBlockSectionNames(false), SeparateNamedSections(false),
        TrapUnreachable(false), NoTrapAfterNoreturn(false), TLSSize(0),
        EmulatedTLS(false), EnableTLSDESC(false), EnableIPRA(false),
        EmitStackSizeSection(false), EnableMachineOutliner(false),
        EnableStaticDataPartitioning(false), EnableDefaultMachineVerifier(true),
        EmitAddrsig(false), BBAddrMap(false), EmitCallGraphSection(false),
        EmitCallSiteInfo(false), EnableDebugEntryValues(false),
        ValueTrackingVariableLocations(false), ForceDwarfFrameSection(false),
        XRayFunctionIndex(true), DebugStrictDwarf(false), JMCInstrument(false),
        EnableCFIFixup(false), MisExpect(false), XCOFFReadOnlyPointers(false),
        VerifyArgABICompliance(false) {}

#define LLVM_TARGET_OPTION_FIELD(Type, Name, Initializer) Type Name Initializer;
#define LLVM_TARGET_OPTION_BITFIELD(Name, Width, WireType)                     \
  unsigned Name : Width;
#define LLVM_TARGET_OPTION_FIRST_BITFIELD LLVM_TARGET_OPTION_BITFIELD
#define LLVM_TARGET_OPTION_MC_FIELD(Name) MCTargetOptions Name;
#include "llvm/Target/TargetOptionsFields.def"
#undef LLVM_TARGET_OPTION_MC_FIELD
#undef LLVM_TARGET_OPTION_FIRST_BITFIELD
#undef LLVM_TARGET_OPTION_BITFIELD
#undef LLVM_TARGET_OPTION_FIELD
};

} // namespace llvm

#endif
