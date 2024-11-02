//===--  riscv.h  - Generic JITLink riscv edge kinds, utilities -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Generic utilities for graphs representing riscv objects.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_JITLINK_RISCV_H
#define LLVM_EXECUTIONENGINE_JITLINK_RISCV_H

#include "llvm/ExecutionEngine/JITLink/JITLink.h"

namespace llvm {
namespace jitlink {
namespace riscv {

/// Represets riscv fixups
enum EdgeKind_riscv : Edge::Kind {

  // TODO: Capture and replace to generic fixups
  /// A plain 32-bit pointer value relocation
  ///
  /// Fixup expression:
  ///   Fixup <= Target + Addend : uint32
  ///
  R_RISCV_32 = Edge::FirstRelocation,

  /// A plain 64-bit pointer value relocation
  ///
  /// Fixup expression:
  ///   Fixup <- Target + Addend : uint32
  ///
  R_RISCV_64,

  /// PC-relative branch pointer value relocation
  ///
  /// Fixup expression:
  ///   Fixup <- (Target - Fixup + Addend)
  ///
  R_RISCV_BRANCH,

  /// High 20 bits of PC-relative jump pointer value relocation
  ///
  /// Fixup expression:
  ///   Fixup <- Target - Fixup + Addend
  ///
  R_RISCV_JAL,

  /// High 20 bits of 32-bit pointer value relocation
  ///
  /// Fixup expression
  ///   Fixup <- (Target + Addend + 0x800) >> 12
  R_RISCV_HI20,

  /// Low 12 bits of 32-bit pointer value relocation, used by S type instruction
  /// format
  ///
  /// Fixup expression
  ///   Fixup <- (Target + Addend) & 0xFFF
  R_RISCV_LO12_S,

  /// Low 12 bits of 32-bit pointer value relocation
  ///
  /// Fixup expression
  ///   Fixup <- (Target + Addend) & 0xFFF
  R_RISCV_LO12_I,
  /// High 20 bits of PC relative relocation
  ///
  /// Fixup expression:
  ///   Fixup <- (Target - Fixup + Addend + 0x800) >> 12
  R_RISCV_PCREL_HI20,

  /// Low 12 bits of PC relative relocation, used by I type instruction format
  ///
  /// Fixup expression:
  ///   Fixup <- (Target - Fixup + Addend) & 0xFFF
  R_RISCV_PCREL_LO12_I,

  /// Low 12 bits of PC relative relocation, used by S type instruction format
  ///
  /// Fixup expression:
  ///   Fixup <- (Target - Fixup + Addend) & 0xFFF
  R_RISCV_PCREL_LO12_S,

  /// PC relative call
  ///
  /// Fixup expression:
  ///   Fixup <- (Target - Fixup + Addend)
  R_RISCV_CALL,

  /// 32 bits PC relative relocation
  ///
  /// Fixup expression:
  ///   Fixup <- (Target - Fixup + Addend)
  R_RISCV_32_PCREL,

  /// PC relative GOT offset
  ///
  /// Fixup expression:
  ///   Fixup <- (GOT - Fixup + Addend) >> 12
  R_RISCV_GOT_HI20,

  /// PC relative call by PLT
  ///
  /// Fixup expression:
  ///   Fixup <- (Target - Fixup + Addend)
  R_RISCV_CALL_PLT,

  /// 64 bits label addition
  ///
  /// Fixup expression:
  ///   Fixup <- (Target - *{8}Fixup + Addend)
  R_RISCV_ADD64,

  /// 32 bits label addition
  ///
  /// Fixup expression:
  ///   Fixup <- (Target - *{4}Fixup + Addend)
  R_RISCV_ADD32,

  /// 16 bits label addition
  ///
  /// Fixup expression
  ///   Fixup <- (Target - *{2}Fixup + Addend)
  R_RISCV_ADD16,

  /// 8 bits label addition
  ///
  /// Fixup expression
  ///   Fixup <- (Target - *{1}Fixup + Addend)
  R_RISCV_ADD8,

  /// 64 bits label subtraction
  ///
  /// Fixup expression
  ///   Fixup <- (Target - *{8}Fixup - Addend)
  R_RISCV_SUB64,

  /// 32 bits label subtraction
  ///
  /// Fixup expression
  ///   Fixup <- (Target - *{4}Fixup - Addend)
  R_RISCV_SUB32,

  /// 16 bits label subtraction
  ///
  /// Fixup expression
  ///   Fixup <- (Target - *{2}Fixup - Addend)
  R_RISCV_SUB16,

  /// 8 bits label subtraction
  ///
  /// Fixup expression
  ///   Fixup <- (Target - *{1}Fixup - Addend)
  R_RISCV_SUB8,

  /// 6 bits label subtraction
  ///
  /// Fixup expression
  ///   Fixup <- (Target - *{1}Fixup - Addend)
  R_RISCV_SUB6,

  /// Local label assignment
  ///
  /// Fixup expression:
  ///   Fixup <- (Target + Addend)
  R_RISCV_SET6,

  /// Local label assignment
  ///
  /// Fixup expression:
  ///   Fixup <- (Target + Addend)
  R_RISCV_SET8,

  /// Local label assignment
  ///
  /// Fixup expression:
  ///   Fixup <- (Target + Addend)
  R_RISCV_SET16,

  /// Local label assignment
  ///
  /// Fixup expression:
  ///   Fixup <- (Target + Addend)
  R_RISCV_SET32,
};

/// Returns a string name for the given riscv edge. For debugging purposes
/// only
const char *getEdgeKindName(Edge::Kind K);
} // namespace riscv
} // namespace jitlink
} // namespace llvm

#endif
