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

/// Mode of ASan detect stack use after return
enum class AsanDetectStackUseAfterReturnMode {
  Never,   ///< Never detect stack use after return.
  Runtime, ///< Detect stack use after return if runtime flag is enabled
           ///< (ASAN_OPTIONS=detect_stack_use_after_return=1)
  Always,  ///< Always detect stack use after return.
  Invalid, ///< Not a valid detect mode.
};

} // namespace llvm

#endif
