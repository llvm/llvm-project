//===--------- Definition of the AddressSanitizer options -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// This file defines data types used to set Address Sanitizer options.
//===----------------------------------------------------------------------===//
#ifndef LLVM_TRANSFORMS_INSTRUMENTATION_ADDRESSSANITIZEROPTIONS_H
#define LLVM_TRANSFORMS_INSTRUMENTATION_ADDRESSSANITIZEROPTIONS_H

namespace llvm {

/// Types of ASan module destructors supported
enum class AsanDtorKind {
  None,    ///< Do not emit any destructors for ASan
  Global,  ///< Append to llvm.global_dtors
  Invalid, ///< Not a valid destructor Kind.
  // TODO(dliew): Add more more kinds.
};
} // namespace llvm
#endif
