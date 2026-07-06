//===---------- llvm/unittests/Target/AMDGPU/AMDGPUUnitTests.h ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_UNITTESTS_TARGET_AMDGPU_AMDGPUUNITTESTS_H
#define LLVM_UNITTESTS_TARGET_AMDGPU_AMDGPUUNITTESTS_H

#include "AMDGPUGenSubtargetInfo.inc"
#include "AMDGPUTargetMachine.h"
#include "CodeGenTestBase.h"
#include "GCNSubtarget.h"
#include <memory>
#include <string>

namespace llvm {
class GCNTargetMachine;
class StringRef;
} // end namespace llvm

std::unique_ptr<llvm::GCNTargetMachine>
createAMDGPUTargetMachine(std::string TStr, llvm::StringRef CPU,
                          llvm::StringRef FS);

class AMDGPUTestBase : public testing::Test {
public:
  static void SetUpTestSuite();
};

class AMDGPUCodeGenTestBase : public llvm::CodeGenTestBase {
public:
  static void SetUpTestSuite();
};

#endif // LLVM_UNITTESTS_TARGET_AMDGPU_AMDGPUUNITTESTS_H
