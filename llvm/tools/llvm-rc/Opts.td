include "llvm/Option/OptParser.td"

// All the switches can be preceded by either '/' or '-'.
// These options seem to be important for the tool
// and should be implemented.

def FILEOUT : Separate<[ "/", "-" ], "FO">,
              HelpText<"Change the output file location.">;

def DEFINE : Separate<[ "/", "-" ], "D">,
             HelpText<"Define a symbol for the C preprocessor.">;
def UNDEF : Separate<[ "/", "-" ], "U">,
            HelpText<"Undefine a symbol for the C preprocessor.">;

def LANG_ID : Separate<[ "/", "-" ], "L">,
              HelpText<"Set the default language identifier.">;
def LANG_NAME : Separate<[ "/", "-" ], "LN">,
                HelpText<"Set the default language name.">;

def INCLUDE : Separate<[ "/", "-" ], "I">, HelpText<"Add an include path.">;
def NOINCLUDE : Flag<[ "/", "-" ], "X">, HelpText<"Ignore 'include' variable.">;

def ADD_NULL : Flag<[ "/", "-" ], "N">,
               HelpText<"Null-terminate all strings in the string table.">;

def DUPID_NOWARN : Flag<[ "/", "-" ], "Y">,
                   HelpText<"Suppress warnings on duplicate resource IDs.">;

def VERBOSE : Flag<[ "/", "-" ], "V">, HelpText<"Be verbose.">;
def HELP : Flag<[ "/", "-" ], "?">, HelpText<"Display this help and exit.">;
def H : Flag<[ "/", "-" ], "H">,
        Alias<HELP>,
        HelpText<"Display this help and exit.">;

def DRY_RUN : Flag<[ "/", "-" ], "dry-run">,
              HelpText<"Don't compile the input; only try to parse it.">;

// Unused switches (at least for now). These will stay unimplemented
// in an early stage of development and can be ignored. However, we need to
// parse them in order to preserve the compatibility with the original tool.

def NOLOGO : Flag<[ "/", "-" ], "NOLOGO">;
def R : Flag<[ "/", "-" ], "R">;
def SL : Flag<[ "/", "-" ], "SL">;

// (Codepages support.)
def C : Flag<[ "/", "-" ], "C">;
def W : Flag<[ "/", "-" ], "W">;

// (Support of MUI and similar.)
def FM : Separate<[ "/", "-" ], "FM">;
def Q : Separate<[ "/", "-" ], "Q">;
def G : Flag<[ "/", "-" ], "G">;
def GN : Flag<[ "/", "-" ], "GN">;
def G1 : Flag<[ "/", "-" ], "G1">;
def G2 : Flag<[ "/", "-" ], "G2">;
