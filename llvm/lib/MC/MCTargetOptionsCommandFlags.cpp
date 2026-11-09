//===-- MCTargetOptionsCommandFlags.cpp -----------------------*- C++ //-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains machine code-specific flags that are shared between
// different command line tools.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCTargetOptionsCommandFlags.h"
#include "llvm/MC/MCTargetOptions.h"
#include "llvm/Option/LibraryOptions.h"

using namespace llvm;

#define OPTIONS_STRUCT_DECL
#include "MCToolOptions.inc"

#define OPTIONS_STRUCT_DEFS
#include "MCToolOptions.inc"

llvm::mc::RegisterMCToolOptions::RegisterMCToolOptions() {
  static opt::RegisterLibraryOptions<MCToolOptions> Registration;
}

StringRef llvm::mc::getABIName(const cl::LibraryOptionsContext &C) {
  return C.get<MCToolOptions>().target_abi;
}

MCTargetOptions
llvm::mc::getMCTargetOptions(const cl::LibraryOptionsContext &C) {
  const MCToolOptions &Opts = C.get<MCToolOptions>();
  MCTargetOptions Options;
  Options.MCRelaxAll = Opts.mc_relax_all;
  Options.MCIncrementalLinkerCompatible = Opts.incremental_linker_compatible;
  Options.FDPIC = Opts.fdpic;
  Options.Dwarf64 = Opts.dwarf64;
  Options.DwarfVersion = Opts.dwarf_version;
  Options.ShowMCInst = Opts.asm_show_inst;
  Options.ABIName = Opts.target_abi;
  Options.MCFatalWarnings = Opts.fatal_warnings;
  Options.MCNoWarn = Opts.no_warn;
  Options.MCNoDeprecatedWarn = Opts.no_deprecated_warn;
  Options.MCNoTypeCheck = Opts.no_type_check;
  Options.MCSaveTempLabels = Opts.save_temp_labels;
  Options.Crel = Opts.crel;
  Options.ImplicitMapSyms = Opts.implicit_mapsyms;
  Options.X86RelaxRelocations = Opts.x86_relax_relocations;
  Options.X86Sse2Avx = Opts.x86_sse2avx;
  Options.DisableIntegratedAS = Opts.no_integrated_as;
  Options.RelocSectionSym = Opts.reloc_section_sym;
  Options.LargeEHEncoding = Opts.large_eh_encoding;
  Options.EmitDwarfUnwind = Opts.emit_dwarf_unwind;
  Options.EmitCompactUnwindNonCanonical =
      Opts.emit_compact_unwind_non_canonical;
  Options.EmitSFrameUnwind = Opts.gsframe;
  Options.AsSecureLogFile = Opts.as_secure_log_file;

  return Options;
}
