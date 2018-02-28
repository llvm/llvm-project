include "llvm/Option/OptParser.td"

def DEFINE : Joined<["/"], "DEFINE:">, HelpText<"">, MetaVarName<"symbol">;
def FOLDDUPS : Flag<["/"], "FOLDDUPS:">, HelpText<"">;
def MACHINE : Joined<["/"], "MACHINE:">, HelpText<"">, MetaVarName<"{ARM|ARM64|EBC|IA64|X64|X86}">;
def NOLOGO : Flag<["/"], "NOLOGO">, HelpText<"">;
def OUT : Joined<["/"], "OUT:">, HelpText<"">, MetaVarName<"filename">;
def READONLY : Flag<["/"], "READONLY">, HelpText<"">;
def VERBOSE : Flag<["/"], "VERBOSE">, HelpText<"">;
def HELP : Flag<["/"], "HELP">;
def H : Flag<["/"], "H">, Alias<HELP>;
