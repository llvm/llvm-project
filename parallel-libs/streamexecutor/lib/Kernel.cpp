//===-- Kernel.cpp - General kernel implementation ------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the implementation details for kernel types.
///
//===----------------------------------------------------------------------===//

#include "streamexecutor/Kernel.h"
#include "streamexecutor/Device.h"
#include "streamexecutor/PlatformInterfaces.h"

#include "llvm/DebugInfo/Symbolize/Symbolize.h"

namespace streamexecutor {

KernelBase::KernelBase(llvm::StringRef Name)
    : Name(Name), DemangledName(llvm::symbolize::LLVMSymbolizer::DemangleName(
                      Name, nullptr)) {}

} // namespace streamexecutor
