//===----------- GDBJITRegistrar.h - GDB JIT interface ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Registration of JIT'd debug objects with debuggers via the GDB JIT
// interface (also implemented by LLDB and other tools).
//
// These functions exist to implement the GDBJITRegistrar allocation actions in
// the SPS controller interface (see sps/GDBJITRegistrarSPSCI.cpp).
//
//===----------------------------------------------------------------------===//

#ifndef ORC_RT_INTERNAL_BEDROCK_GDBJITREGISTRAR_H
#define ORC_RT_INTERNAL_BEDROCK_GDBJITREGISTRAR_H

#include "orc-rt/support/Error.h"
#include "orc-rt/support/span.h"

namespace orc_rt::gdb_jit {

/// Register the object in the given buffer with the GDB JIT interface.
///
/// Obj range must remain mapped and unmodified until it is deregistered:
/// debuggers read the object image out of this range, and this may happen at
/// any time after the registration.
Error registerObject(span<char> Obj);

/// Deregister the object in the given buffer from the GDB JIT interface.
///
/// Returns an error if no object is registered for Obj.
///
/// Callers must deregister before releasing Obj's memory, so that a debugger
/// can never observe a list entry describing memory that has been reused.
Error deregisterObject(span<char> Obj);

} // namespace orc_rt::gdb_jit

#endif // ORC_RT_INTERNAL_BEDROCK_GDBJITREGISTRAR_H
