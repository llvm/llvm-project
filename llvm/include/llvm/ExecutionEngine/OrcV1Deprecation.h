//===------ OrcV1Deprecation.h - Memory manager for MC-JIT ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tag for suppressing ORCv1 deprecation warnings.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORCV1DEPRECATION_H
#define LLVM_EXECUTIONENGINE_ORCV1DEPRECATION_H

namespace llvm {

enum ORCv1DeprecationAcknowledgement { AcknowledgeORCv1Deprecation };

} // namespace llvm

#endif // LLVM_EXECUTIONENGINE_ORCV1DEPRECATION_H
