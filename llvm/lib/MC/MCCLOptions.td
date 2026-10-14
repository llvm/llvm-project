//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "llvm/Option/LibraryOptions.td"

def MCCLOptions : OptionsStruct;

defm : DefaultOnOffField<"dwarf-extended-loc",
  "Disable emission of the extended flags in .loc directives">;
defm : BoolField<"lfi-enable-rewriter", "true",
  "Enable rewriting for LFI">;
defm : ValueField<"sched-model-reservation-station-scale-factor", "float", "1.0f",
  "Scale the buffer size of all reservation stations by a positive factor. "
  "Buffer sizes of -1/0/1 (unlimited/unbuffered/in-order) are preserved. Likewise, if the scaled result is <= 1, the original size is kept. Computed sizes are truncated towards zero">;
defm : OptionalBoolField<"use-leb128-directives",
  "Disable the usage of LEB128 directives, and generate .byte instead">;
