//===------------ Program.h - liboffload program abstraction ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Declares ProgramTy, which wraps a liboffload program handle (a device
// image loaded via olCreateProgram) along with the routines used to resolve
// its global-variable and kernel symbols.
//
//===----------------------------------------------------------------------===//

#ifndef _OMPTARGET_PROGRAM_H
#define _OMPTARGET_PROGRAM_H

#include "OffloadAPI.h"
#include "Shared/APITypes.h"
#include "llvm/Support/Error.h"

class ProgramTy {
public:
  ProgramTy() = default;

  /// Load \p Img onto \p Device within \p Context via liboffload.
  static llvm::Expected<ProgramTy> create(ol_context_handle_t Context,
                                          ol_device_handle_t Device,
                                          __tgt_device_image *Img);

  /// Resolve the device address of the global variable \p Name. If \p Size
  /// is non-null, it is set to the size in bytes of the global
  llvm::Expected<void *> getGlobalAddress(const char *Name,
                                          size_t *Size = nullptr) const;

  /// Resolve the opaque plugin kernel handle for the kernel \p Name.
  llvm::Expected<void *> getKernelAddress(const char *Name) const;

  ol_program_handle_t getHandle() const { return Handle; }

private:
  explicit ProgramTy(ol_program_handle_t Handle) : Handle(Handle) {}

  ol_program_handle_t Handle = nullptr;
};

#endif
