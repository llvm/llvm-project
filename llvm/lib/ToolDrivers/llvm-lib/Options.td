include "llvm/Option/OptParser.td"

// lib.exe accepts options starting with either a dash or a slash.

// Flag that takes no arguments.
class F<string name> : Flag<["/", "-", "/?", "-?"], name>;

// Flag that takes one argument after ":".
class P<string name, string help> :
      Joined<["/", "-", "/?", "-?"], name#":">, HelpText<help>;

def libpath: P<"libpath", "Object file search path">;

// Can't be called "list" since that's a keyword.
def lst    : F<"list">, HelpText<"List contents of .lib file on stdout">;
def out    : P<"out", "Path to file to write output">;

def llvmlibthin : F<"llvmlibthin">,
    HelpText<"Make .lib point to .obj files instead of copying their contents">;

def llvmlibempty : F<"llvmlibempty">,
    HelpText<"When given no contents, produce an empty .lib file">;

def machine: P<"machine", "Specify target platform">;

def help : F<"help">;

// /?? and -?? must be before /? and -? to not confuse lib/Options.
def help_q : Flag<["/??", "-??", "/?", "-?"], "">, Alias<help>;

//==============================================================================
// The flags below do nothing. They are defined only for lib.exe compatibility.
//==============================================================================

class QF<string name> : Joined<["/", "-", "/?", "-?"], name#":">;

def ignore : QF<"ignore">;
def nologo : F<"nologo">;
