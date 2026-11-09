//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "llvm/Option/LibraryOptions.td"

def MCToolOptions : OptionsStruct;

defm : ValueField<"as-secure-log-file", "StringRef", "",
  "As secure log file name">;
defm : BoolField<"asm-show-inst", "false",
  "Emit internal instruction representation to assembly file">;
defm : BoolField<"crel", "false",
  "Use CREL relocation format for ELF">;
defm : ValueField<"dwarf-version", "int", "0",
  "Dwarf version">;
defm : BoolField<"dwarf64", "false",
  "Generate debugging info in the 64-bit DWARF format">;
// By default, use DWARF for non-canonical personalities.
defm : BoolField<"emit-compact-unwind-non-canonical", "false",
  "Whether to try to emit Compact Unwind for non canonical entries">;
defm : EnumField<"emit-dwarf-unwind", "EmitDwarfUnwindType", "EmitDwarfUnwindType::Default",
  "always,no-compact-unwind,dwarf-only,default",
  ["EmitDwarfUnwindType::Always", "EmitDwarfUnwindType::NoCompactUnwind", "EmitDwarfUnwindType::DwarfOnly", "EmitDwarfUnwindType::Default"],
  "Whether to emit DWARF EH frame entries">;
defm : BoolField<"fatal-warnings", "false",
  "Treat warnings as errors">;
defm : BoolField<"fdpic", "false",
  "Use the FDPIC ABI">;
defm : BoolField<"gsframe", "false",
  "Whether to emit .sframe unwind sections">;
defm : BoolField<"implicit-mapsyms", "false",
  "Allow mapping symbol at section beginning to be implicit, lowering number of mapping symbols at the expense of some portability. Recommended for projects that can build all their object files using this option">;
defm : BoolField<"incremental-linker-compatible", "false",
  "When used with filetype=obj, emit an object file which can be used with an incremental linker">;
defm : BoolField<"large-eh-encoding", "false",
  "Force 8-byte (sdata8) pointer encodings for ELF exception-handling sections to avoid relocation overflows in large binaries (x86_64: FDE/personality/LSDA/TType; AArch64/PPC64: FDE only, the rest are already sdata8)">;
defm : BoolField<"mc-relax-all", "false",
  "When used with filetype=obj, relax all fixups in the emitted object file">;
defm : BoolField<"no-deprecated-warn", "false",
  "Suppress all deprecated warnings">;
defm : BoolField<"no-integrated-as", "false",
  "Disable integrated assembler">;
defm : BoolField<"no-type-check", "false",
  "Suppress type errors (Wasm)">;
defm : BoolField<"no-warn", "false",
  "Suppress all warnings">;
def : AliasOf<"W", "no-warn">;
defm : EnumField<"reloc-section-sym", "RelocSectionSymType", "RelocSectionSymType::All",
  "all,internal,none",
  ["RelocSectionSymType::All", "RelocSectionSymType::Internal", "RelocSectionSymType::None"],
  "Control section symbol conversion for relocations">;
defm : BoolField<"save-temp-labels", "false",
  "Don't discard temporary labels">;
defm : ValueField<"target-abi", "StringRef", "",
  "The name of the ABI to be targeted from the backend">;
defm : BoolField<"x86-relax-relocations", "true",
  "Emit GOTPCRELX/REX_GOTPCRELX/CODE_4_GOTPCRELX instead of GOTPCREL on x86-64 ELF">;
defm : BoolField<"x86-sse2avx", "false",
  "Specify that the assembler should encode SSE instructions with VEX prefix">;
