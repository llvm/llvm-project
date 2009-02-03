//===- RegisterCoalescer.cpp - Generic Register Coalescing Interface -------==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the generic RegisterCoalescer interface which
// is used as the common interface used by all clients and
// implementations of register coalescing.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/RegisterCoalescer.h"
#include "llvm/CodeGen/LiveIntervalAnalysis.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Pass.h"

using namespace llvm;

// Register the RegisterCoalescer interface, providing a nice name to refer to.
static RegisterAnalysisGroup<RegisterCoalescer> Z("Register Coalescer");
char RegisterCoalescer::ID = 0;

// RegisterCoalescer destructor: DO NOT move this to the header file
// for RegisterCoalescer or else clients of the RegisterCoalescer
// class may not depend on the RegisterCoalescer.o file in the current
// .a file, causing alias analysis support to not be included in the
// tool correctly!
//
RegisterCoalescer::~RegisterCoalescer() {}

// Because of the way .a files work, we must force the SimpleRC
// implementation to be pulled in if the RegisterCoalescer classes are
// pulled in.  Otherwise we run the risk of RegisterCoalescer being
// used, but the default implementation not being linked into the tool
// that uses it.
DEFINING_FILE_FOR(RegisterCoalescer)
