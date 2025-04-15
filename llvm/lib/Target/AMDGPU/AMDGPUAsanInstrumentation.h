//===AMDGPUAsanInstrumentation.h - ASAN helper functions -*- C++- *===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===--------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_UTILS_AMDGPU_ASAN_INSTRUMENTATION_H
#define LLVM_LIB_TARGET_AMDGPU_UTILS_AMDGPU_ASAN_INSTRUMENTATION_H

#include "AMDGPU.h"
#include "AMDGPUMemoryUtils.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/OptimizedStructLayout.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Instrumentation/AddressSanitizer.h"
#include "llvm/Transforms/Instrumentation/AddressSanitizerCommon.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

namespace llvm {
namespace AMDGPU {

/// Given SizeInBytes of the Value to be instrunmented,
/// Returns the redzone size corresponding to it.
uint64_t getRedzoneSizeForGlobal(int Scale, uint64_t SizeInBytes);

/// Instrument the memory operand Addr.
/// Generates report blocks that catch the addressing errors.
void instrumentAddress(Module &M, IRBuilder<> &IRB, Instruction *OrigIns,
                       Instruction *InsertBefore, Value *Addr, Align Alignment,
                       TypeSize TypeStoreSize, bool IsWrite,
                       Value *SizeArgument, bool UseCalls, bool Recover,
                       int Scale, int Offset);

/// Get all the memory operands from the instruction
/// that needs to be instrumented
void getInterestingMemoryOperands(
    Module &M, Instruction *I,
    SmallVectorImpl<InterestingMemoryOperand> &Interesting);

} // end namespace AMDGPU
} // end namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_UTILS_AMDGPU_ASAN_INSTRUMENTATION_H
