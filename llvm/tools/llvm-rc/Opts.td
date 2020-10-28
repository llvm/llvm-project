include "llvm/Option/OptParser.td"

// All the switches can be preceded by either '/' or '-'.
// These options seem to be important for the tool
// and should be implemented.

def fileout : JoinedOrSeparate<[ "/", "-" ], "FO">,
              HelpText<"Change the output file location.">;

def define : Separate<[ "/", "-" ], "D">,
             HelpText<"Define a symbol for the C preprocessor.">;
def undef : Separate<[ "/", "-" ], "U">,
            HelpText<"Undefine a symbol for the C preprocessor.">;

def lang_id : JoinedOrSeparate<[ "/", "-" ], "L">,
              HelpText<"Set the default language identifier.">;
def lang_name : Separate<[ "/", "-" ], "LN">,
                HelpText<"Set the default language name.">;

def includepath : Separate<[ "/", "-" ], "I">, HelpText<"Add an include path.">;
def noinclude : Flag<[ "/", "-" ], "X">, HelpText<"Ignore 'include' variable.">;

def add_null : Flag<[ "/", "-" ], "N">,
               HelpText<"Null-terminate all strings in the string table.">;

def dupid_nowarn : Flag<[ "/", "-" ], "Y">,
                   HelpText<"Suppress warnings on duplicate resource IDs.">;

def verbose : Flag<[ "/", "-" ], "V">, HelpText<"Be verbose.">;
def help : Flag<[ "/", "-" ], "?">, HelpText<"Display this help and exit.">;
def h : Flag<[ "/", "-" ], "H">,
        Alias<help>,
        HelpText<"Display this help and exit.">;

def dry_run : Flag<[ "/", "-" ], "dry-run">,
              HelpText<"Don't compile the input; only try to parse it.">;

def codepage : JoinedOrSeparate<[ "/", "-" ], "C">,
               HelpText<"Set the codepage used for input strings.">;

// Unused switches (at least for now). These will stay unimplemented
// in an early stage of development and can be ignored. However, we need to
// parse them in order to preserve the compatibility with the original tool.

def nologo : Flag<[ "/", "-" ], "NOLOGO">;
def r : Flag<[ "/", "-" ], "R">;
def sl : Flag<[ "/", "-" ], "SL">;

// (Codepages support.)
def w : Flag<[ "/", "-" ], "W">;

// (Support of MUI and similar.)
def fm : Separate<[ "/", "-" ], "FM">;
def q : Separate<[ "/", "-" ], "Q">;
def g : Flag<[ "/", "-" ], "G">;
def gn : Flag<[ "/", "-" ], "GN">;
def g1 : Flag<[ "/", "-" ], "G1">;
def g2 : Flag<[ "/", "-" ], "G2">;
