//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "llvm/Option/LibraryOptions.td"

def AArch64MCOptions : OptionsStruct<prefix = "aarch64-">;

// AArch64MCAsmInfo.cpp
defm : EnumField<"aarch64-neon-syntax", "std::optional<AArch64::AsmWriterVariantTy>", "std::nullopt",
  "generic,apple", ["AArch64::Generic", "AArch64::Apple"],
  "Choose style of NEON code to emit from AArch64 backend:">;

// AArch64MCLFIRewriter.cpp
defm : BoolField<"aarch64-lfi-guard-elim", "true",
  "Enable the LFI guard elimination optimization">;

// AArch64TargetStreamer.cpp
defm : BoolField<"aarch64-mark-bti-property", "false",
  "Add .note.gnu.property with BTI to assembly files">;
