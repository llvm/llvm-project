//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "llvm/Option/LibraryOptions.td"

def RISCVMCOptions : OptionsStruct<prefix = "riscv-">;

// AsmParser/RISCVAsmParser.cpp
defm : BoolField<"riscv-add-build-attributes", "false", "">;

// MCTargetDesc/RISCVAsmBackend.cpp
defm : BoolField<"riscv-align-rvc", "true",
  "When generating R_RISCV_ALIGN, insert $alignment-2 bytes of NOPs even in norvc code">;
// Temporary workaround for old linkers that do not support ULEB128 relocations,
// which are abused by DWARF v5 DW_LLE_offset_pair/DW_RLE_offset_pair
// implemented in Clang/LLVM.
defm : BoolField<"riscv-uleb128-reloc", "true",
  "Emit R_RISCV_SET_ULEB128/E_RISCV_SUB_ULEB128 if appropriate">;

// MCTargetDesc/RISCVInstPrinter.cpp
defm : BoolField<"riscv-emit-x8-as-fp", "false",
  "Emit x8 as fp instead of s0">;
defm : BoolField<"riscv-no-aliases", "false",
  "Disable the emission of assembler pseudo instructions">;

// MCTargetDesc/RISCVTargetStreamer.cpp
// This option controls whether or not we emit ELF attributes for ABI features,
// like RISC-V atomics or X3 usage.
defm : BoolField<"riscv-abi-attributes", "false",
  "Enable emitting RISC-V ELF attributes for ABI features">;
