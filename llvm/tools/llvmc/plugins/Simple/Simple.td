// A simple wrapper for gcc.
// To compile, use this command:
//
//      $ cd $LLVMC2_DIR
//      $ make DRIVER_NAME=mygcc BUILTIN_PLUGINS=Simple
//
// To build this plugin as a dynamic library:
//
//      $ cd $LLVMC2_DIR
//      $ make BUILTIN_PLUGINS=""
//      $ cd plugins/Simple
//      $ make
//
// Run as:
//
//      $ llvmc2 -load $LLVM_DIR/Release/lib/LLVMCSimple.so

include "llvm/CompilerDriver/Common.td"

def gcc : Tool<
[(in_language "c"),
 (out_language "executable"),
 (output_suffix "out"),
 (cmd_line "gcc $INFILE -o $OUTFILE"),
 (sink)
]>;

def LanguageMap : LanguageMap<[LangToSuffixes<"c", ["c"]>]>;

def CompilationGraph : CompilationGraph<[Edge<"root", "gcc">]>;
