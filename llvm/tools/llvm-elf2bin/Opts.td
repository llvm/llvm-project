//===- Opts.td - llvm-elf2bin's command-line option definitions -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "llvm/Option/OptParser.td"

multiclass Value<string longname, string shortname, string metavar,
                 string help, OptionGroup group> {
  def NAME # _EQ : Joined<["--"], longname # "=">,
                   HelpText<help>,
                   MetaVarName<metavar>,
                   Group<group>;
  def : Separate<["--"], longname>,
        Alias<!cast<Joined>(NAME # "_EQ")>;

  if !not(!empty(shortname)) then {
    def : JoinedOrSeparate<["-"], shortname>,
          Alias<!cast<Joined>(NAME # "_EQ")>,
          HelpText<"Alias for --" # longname>, Group<group>;
  }
}

multiclass NoValue<string longname, string shortname, string help,
                   OptionGroup group> {
  def NAME : Flag<["--"], longname>,
             HelpText<help>,
             Group<group>;

  if !not(!empty(shortname)) then {
    def : Flag<["-"], shortname>,
          Alias<!cast<Flag>(NAME)>,
          HelpText<"Alias for --" # longname>;
  }
}

def grp_mode: OptionGroup<"mode">, HelpText<"FORMAT OF OUTPUT">;
def grp_file: OptionGroup<"output">, HelpText<"LOCATION OF OUTPUT">;
def grp_opts: OptionGroup<"options">, HelpText<"OPTIONS">;

defm output_file: Value<"output-file", "o", "FILENAME", "Name of the output file. Invalid if more than one output file is to be generated.", grp_file>;
defm output_pattern: Value<"output-pattern", "O", "PATTERN", "Schema for naming all output files. May contain %f for base name of input file; %F for full name of input file; %a or %A for base address of ELF segment, in hex or in HEX; %b for bank number, if using --banks; %% for a literal % sign.", grp_file>;

defm ihex: NoValue<"ihex", "", "Output Intel Hex files (\":108000\" style)", grp_mode>;
defm srec: NoValue<"srec", "", "Output Motorola Hex files (\"S3150800\" style)", grp_mode>;
defm bin: NoValue<"bin", "", "Output a binary file per ELF segment", grp_mode>;
defm bincombined: NoValue<"bincombined", "", "Output a single binary file, containing all segments, with padding to place them at the correct relative positions", grp_mode>;
defm vhx: NoValue<"vhx", "", "Output a Verilog Hex file per ELF segment", grp_mode>;
defm vhxcombined: NoValue<"vhxcombined", "", "Output a single Verilog Hex file, containing all segments, with padding to place them at the correct relative positions", grp_mode>;

defm base: Value<"base", "", "ADDRESS", "Base address of the whole output file, for --bincombined or --vhxcombined", grp_opts>;
defm banks: Value<"banks", "", "WIDTHxNUM", "Partition the output into subfiles by writing WIDTH bytes in turn to each of NUM files in cyclic order, for binary or Verilog Hex output", grp_opts>;
defm datareclen: Value<"datareclen", "", "LENGTH", "Number of data bytes to put into each record, for Intel Hex or Motorola Hex output", grp_opts>;
defm segments: Value<"segments", "", "ADDRESS,ADDRESS,...", "Select only the ELF segments of the input which start at the specified addresses", grp_opts>;
defm zi: NoValue<"zi", "", "Include the zero-initialized padding at the end of each ELF segment", grp_opts>;
defm physical: NoValue<"physical", "", "Use the physical addresses (p_paddr) of ELF segments", grp_opts>;
defm virtual: NoValue<"virtual", "", "Use the virtual addresses (p_vaddr) of ELF segments", grp_opts>;

defm help: NoValue<"help", "", "Display this help", grp_opts>;
defm version: NoValue<"version", "", "Display the version", grp_opts>;
