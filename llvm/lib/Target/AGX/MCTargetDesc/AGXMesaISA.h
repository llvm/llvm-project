//===-- AGXMesaISA.h - Mesa's AGX instruction forms -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The instruction forms Mesa's asahi compiler emits for G13 and G14, and a
// check that an encoded instruction is one of them. The forms are generated
// by llvm/utils/agx-mesa-isa/mesa_isa.py, which documents them.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AGX_MCTARGETDESC_AGXMESAISA_H
#define LLVM_LIB_TARGET_AGX_MCTARGETDESC_AGXMESAISA_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include <optional>
#include <string>

namespace llvm {

class MCSubtargetInfo;

namespace AGX {

/// The device variants Mesa packs differently: only bit 45 of device atomics
/// differs, set for multi-cluster G13 and for two-die parts.
enum class MesaGen { G13, G13X, G14, G14D };

/// The Mesa variant of \p STI's processor, or std::nullopt when the processor
/// is not a G13 or G14.
std::optional<MesaGen> getMesaGen(const MCSubtargetInfo &STI);

/// Whether \p Bytes (one whole instruction) is a form Mesa emits for \p Gen.
/// On success \p Form names the form; otherwise \p Why says what differs from
/// the closest form.
bool isMesaInstruction(ArrayRef<uint8_t> Bytes, MesaGen Gen,
                       StringRef *Form = nullptr, std::string *Why = nullptr);

} // namespace AGX
} // namespace llvm

#endif
