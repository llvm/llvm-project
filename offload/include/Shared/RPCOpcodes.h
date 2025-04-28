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
  OFFLOAD_EMISSARY = LLVM_OFFLOAD_OPCODE(1),
  EMISSARY_PREMALLOC = LLVM_OFFLOAD_OPCODE(2),
  EMISSARY_FREE = LLVM_OFFLOAD_OPCODE(3),
  ALT_LIBC_MALLOC = LLVM_OFFLOAD_OPCODE(4),
  ALT_LIBC_FREE = LLVM_OFFLOAD_OPCODE(5),
} offload_opcode_t;

#undef LLVM_OFFLOAD_OPCODE

#endif // OMPTARGET_SHARED_RPC_OPCODES_H
