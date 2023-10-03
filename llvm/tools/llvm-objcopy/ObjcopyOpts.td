include "CommonOpts.td"

defm binary_architecture
    : Eq<"binary-architecture", "Ignored for compatibility">;
def B : JoinedOrSeparate<["-"], "B">,
        Alias<binary_architecture>,
        HelpText<"Alias for --binary-architecture">;

defm target : Eq<"target", "Format of the input and output file">,
              Values<"binary">;
def F : JoinedOrSeparate<["-"], "F">,
        Alias<target>,
        HelpText<"Alias for --target">;

defm input_target : Eq<"input-target", "Format of the input file">,
                    Values<"binary">;
def I : JoinedOrSeparate<["-"], "I">,
        Alias<input_target>,
        HelpText<"Alias for --input-target">;

defm output_target : Eq<"output-target", "Format of the output file">,
                     Values<"binary">;
def O : JoinedOrSeparate<["-"], "O">,
        Alias<output_target>,
        HelpText<"Alias for --output-target">;

defm new_symbol_visibility : Eq<"new-symbol-visibility", "Visibility of "
                                "symbols generated for binary input or added"
                                " with --add-symbol unless otherwise"
                                " specified. The default value is 'default'">;

def compress_debug_sections
    : Joined<["--"], "compress-debug-sections=">,
      MetaVarName<"format">,
      HelpText<"Compress DWARF debug sections using specified format. Supported "
               "formats: zlib, zstd. Select zlib if <format> is omitted">;
def : Flag<["--"], "compress-debug-sections">, Alias<compress_debug_sections>,
      AliasArgs<["zlib"]>;
def decompress_debug_sections : Flag<["--"], "decompress-debug-sections">,
                                HelpText<"Decompress DWARF debug sections">;
defm split_dwo
    : Eq<"split-dwo", "Equivalent to extract-dwo on the input file to "
                      "<dwo-file>, then strip-dwo on the input file">,
      MetaVarName<"dwo-file">;

defm add_gnu_debuglink
    : Eq<"add-gnu-debuglink", "Add a .gnu_debuglink for <debug-file>">,
      MetaVarName<"debug-file">;

defm rename_section
    : Eq<"rename-section",
         "Renames a section from old to new, optionally with specified flags. "
         "Flags supported for GNU compatibility: alloc, load, noload, "
         "readonly, exclude, debug, code, data, rom, share, contents, merge, "
         "strings, large">,
      MetaVarName<"old=new[,flag1,...]">;
defm redefine_symbol
    : Eq<"redefine-sym", "Change the name of a symbol old to new">,
      MetaVarName<"old=new">;
defm redefine_symbols
    : Eq<"redefine-syms",
         "Reads a list of symbol pairs from <filename> and runs as if "
         "--redefine-sym=<old>=<new> is set for each one. <filename> "
         "contains two symbols per line separated with whitespace and may "
         "contain comments beginning with '#'. Leading and trailing "
         "whitespace is stripped from each line. May be repeated to read "
         "symbols from many files">,         
      MetaVarName<"filename">;

defm only_section : Eq<"only-section", "Remove all but <section>">,
                    MetaVarName<"section">;
def j : JoinedOrSeparate<["-"], "j">,
        Alias<only_section>,
        HelpText<"Alias for --only-section">;
defm add_section
    : Eq<"add-section",
         "Make a section named <section> with the contents of <file>">,
      MetaVarName<"section=file">;

defm set_section_alignment
    : Eq<"set-section-alignment", "Set alignment for a given section">,
      MetaVarName<"section=align">;

defm set_section_flags
    : Eq<"set-section-flags",
         "Set section flags for a given section. Flags supported for GNU "
         "compatibility: alloc, load, noload, readonly, exclude, debug, code, "
         "data, rom, share, contents, merge, strings, large">,
      MetaVarName<"section=flag1[,flag2,...]">;

defm set_section_type
    : Eq<"set-section-type",
         "Set the type of section <section> to the integer <type>">,
      MetaVarName<"section=type">;

def S : Flag<["-"], "S">,
        Alias<strip_all>,
        HelpText<"Alias for --strip-all">;
def strip_dwo : Flag<["--"], "strip-dwo">,
                HelpText<"Remove all DWARF .dwo sections from file">;
def strip_non_alloc
    : Flag<["--"], "strip-non-alloc">,
      HelpText<"Remove all non-allocated sections outside segments">;
defm strip_unneeded_symbol
    : Eq<"strip-unneeded-symbol",
         "Remove symbol <symbol> if it is not needed by relocations">,
      MetaVarName<"symbol">;
defm strip_unneeded_symbols
    : Eq<"strip-unneeded-symbols",
         "Reads a list of symbols from <filename> and removes them "
         "if they are not needed by relocations">,
      MetaVarName<"filename">;

defm subsystem
    : Eq<"subsystem",
         "Set PE subsystem and version">,
      MetaVarName<"name[:version]">;

def extract_dwo
    : Flag<["--"], "extract-dwo">,
      HelpText<
          "Remove all sections that are not DWARF .dwo sections from file">;

defm extract_partition
    : Eq<"extract-partition", "Extract named partition from input file">,
      MetaVarName<"name">;
def extract_main_partition
    : Flag<["--"], "extract-main-partition">,
      HelpText<"Extract main partition from the input file">;

def localize_hidden
    : Flag<["--"], "localize-hidden">,
      HelpText<
          "Mark all symbols that have hidden or internal visibility as local">;
defm localize_symbol : Eq<"localize-symbol", "Mark <symbol> as local">,
                       MetaVarName<"symbol">;
defm localize_symbols
    : Eq<"localize-symbols",
         "Reads a list of symbols from <filename> and marks them local">,
      MetaVarName<"filename">;

def L : JoinedOrSeparate<["-"], "L">,
        Alias<localize_symbol>,
        HelpText<"Alias for --localize-symbol">;

defm globalize_symbol : Eq<"globalize-symbol", "Mark <symbol> as global">,
                        MetaVarName<"symbol">;

defm globalize_symbols
    : Eq<"globalize-symbols",
         "Reads a list of symbols from <filename> and marks them global">,
      MetaVarName<"filename">;

defm keep_global_symbol
    : Eq<"keep-global-symbol",
         "Convert all symbols except <symbol> to local. May be repeated to "
         "convert all except a set of symbols to local">,
      MetaVarName<"symbol">;
def G : JoinedOrSeparate<["-"], "G">,
        Alias<keep_global_symbol>,
        HelpText<"Alias for --keep-global-symbol">;

defm keep_global_symbols
    : Eq<"keep-global-symbols",
         "Reads a list of symbols from <filename> and runs as if "
         "--keep-global-symbol=<symbol> is set for each one. <filename> "
         "contains one symbol per line and may contain comments beginning with "
         "'#'. Leading and trailing whitespace is stripped from each line. May "
         "be repeated to read symbols from many files">,
      MetaVarName<"filename">;

defm weaken_symbol : Eq<"weaken-symbol", "Mark <symbol> as weak">,
                     MetaVarName<"symbol">;
defm weaken_symbols
    : Eq<"weaken-symbols",
         "Reads a list of symbols from <filename> and marks them weak">,
      MetaVarName<"filename">;

def W : JoinedOrSeparate<["-"], "W">,
        Alias<weaken_symbol>,
        HelpText<"Alias for --weaken-symbol">;
def weaken : Flag<["--"], "weaken">,
             HelpText<"Mark all global symbols as weak">;

defm strip_symbols
    : Eq<"strip-symbols",
         "Reads a list of symbols from <filename> and removes them">,
      MetaVarName<"filename">;

defm keep_symbols
    : Eq<"keep-symbols",
         "Reads a list of symbols from <filename> and runs as if "
         "--keep-symbol=<symbol> is set for each one. <filename> "
         "contains one symbol per line and may contain comments beginning with "
         "'#'. Leading and trailing whitespace is stripped from each line. May "
         "be repeated to read symbols from many files">,
      MetaVarName<"filename">;

defm dump_section
    : Eq<"dump-section",
         "Dump contents of section named <section> into file <file>">,
      MetaVarName<"section=file">;
defm prefix_symbols
    : Eq<"prefix-symbols", "Add <prefix> to the start of every symbol name">,
      MetaVarName<"prefix">;

defm prefix_alloc_sections
    : Eq<"prefix-alloc-sections", "Add <prefix> to the start of every allocated section name">,
      MetaVarName<"prefix">;

defm set_start : Eq<"set-start", "Set the start address to <addr>. Overrides "
                    "any previous --change-start or --adjust-start values">,
                 MetaVarName<"addr">;
defm change_start : Eq<"change-start", "Add <incr> to the start address. Can be "                        
                       "specified multiple times, all values will be applied "
                       "cumulatively">,
                    MetaVarName<"incr">;
def adjust_start : JoinedOrSeparate<["--"], "adjust-start">,
                   Alias<change_start>,
                   HelpText<"Alias for --change-start">;

defm add_symbol
    : Eq<"add-symbol", "Add new symbol <name> to .symtab. Accepted flags: "
         "global, local, weak, default, hidden, protected, file, section, object, "
         "function, indirect-function. Accepted but ignored for "
         "compatibility: debug, constructor, warning, indirect, synthetic, "
         "unique-object, before">,
      MetaVarName<"name=[section:]value[,flags]">;

defm update_section
    : Eq<"update-section", "Replace the contents of section <name> with contents from a file <file>">,
      MetaVarName<"name=file">;
