//===-- Shared/RPCOpcodes.h - Offload specific RPC opcodes ----- C++ ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines RPC opcodes that are specifically used by the OpenMP device runtime.
//
//===----------------------------------------------------------------------===//

#ifndef OMPTARGET_SHARED_RPC_OPCODES_H
#define OMPTARGET_SHARED_RPC_OPCODES_H

#define LLVM_OFFLOAD_RPC_BASE 'o'
#define LLVM_OFFLOAD_OPCODE(n) (LLVM_OFFLOAD_RPC_BASE << 24 | n)

typedef enum {
  OFFLOAD_HOST_CALL = LLVM_OFFLOAD_OPCODE(0),
} offload_opcode_t;

#undef LLVM_OFFLOAD_OPCODE

#endif // OMPTARGET_SHARED_RPC_OPCODES_H
