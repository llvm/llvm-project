//=-- Hexagon.h - Top-level interface for Hexagon representation --*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the entry points for global functions defined in the LLVM
// Hexagon back-end.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_HEXAGON_HEXAGON_H
#define LLVM_LIB_TARGET_HEXAGON_HEXAGON_H

#define Hexagon_POINTER_SIZE 4

#define Hexagon_PointerSize (Hexagon_POINTER_SIZE)
#define Hexagon_PointerSize_Bits (Hexagon_POINTER_SIZE * 8)
#define Hexagon_WordSize Hexagon_PointerSize
#define Hexagon_WordSize_Bits Hexagon_PointerSize_Bits

// allocframe saves LR and FP on stack before allocating
// a new stack frame. This takes 8 bytes.
#define HEXAGON_LRFP_SIZE 8

// Normal instruction size (in bytes).
#define HEXAGON_INSTR_SIZE 4

// Maximum number of words and instructions in a packet.
#define HEXAGON_PACKET_SIZE 4
#define HEXAGON_MAX_PACKET_SIZE (HEXAGON_PACKET_SIZE * HEXAGON_INSTR_SIZE)
// Minimum number of instructions in an end-loop packet.
#define HEXAGON_PACKET_INNER_SIZE 2
#define HEXAGON_PACKET_OUTER_SIZE 3
// Maximum number of instructions in a packet before shuffling,
// including a compound one or a duplex or an extender.
#define HEXAGON_PRESHUFFLE_PACKET_SIZE (HEXAGON_PACKET_SIZE + 3)

// Name of the global offset table as defined by the Hexagon ABI
#define HEXAGON_GOT_SYM_NAME "_GLOBAL_OFFSET_TABLE_"

#include "MCTargetDesc/HexagonMCTargetDesc.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {
  class HexagonTargetMachine;

  /// \brief Creates a Hexagon-specific Target Transformation Info pass.
  ImmutablePass *createHexagonTargetTransformInfoPass(const HexagonTargetMachine *TM);
} // end namespace llvm;

#endif
