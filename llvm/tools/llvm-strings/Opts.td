include "llvm/Option/OptParser.td"

class F<string letter, string help> : Flag<["-"], letter>, HelpText<help>;
class FF<string name, string help> : Flag<["--"], name>, HelpText<help>;

multiclass Eq<string name, string help> {
  def NAME #_EQ : Joined<["--"], name #"=">,
                  HelpText<help>;
  def : Separate<["--"], name>, Alias<!cast<Joined>(NAME #_EQ)>;
}

def all : FF<"all", "Silently ignored. Present for GNU strings compatibility">;
defm bytes : Eq<"bytes", "Print sequences of the specified length">;
defm encoding : Eq<"encoding", [{Select the character encoding of the strings to find:
s (7-bit characters)
S (8-bit characters in the current locale)
u (UTF-8)
b (UTF-16 big endian)
l (UTF-16 little endian)
B (UTF-32 big endian)
L (UTF-32 little endian)}]>;
def help : FF<"help", "Display this help">;
def print_file_name : Flag<["--"], "print-file-name">, HelpText<"Print the name of the file before each string">;
defm radix : Eq<"radix", "Print the offset within the file with the specified radix: o (octal), d (decimal), x (hexadecimal)">, MetaVarName<"<radix>">;
defm unicode : Eq<"unicode", [{Control the display of UTF-8 multibyte characters:
default   (print UTF-8 characters)
invalid   (reject multibyte characters)
locale    (print characters in the current locale)
escape    (\u0000 escape sequences)
hex       (print UTF-8 encoding in hex)
highlight (like escape, but highlighted)}]>;
def version : FF<"version", "Display the version">;

def : F<"a", "Alias for --all">, Alias<all>;
def : F<"f", "Alias for --print-file-name">, Alias<print_file_name>;
def : F<"h", "Alias for --help">, Alias<help>;
def : JoinedOrSeparate<["-"], "n">, Alias<bytes_EQ>, HelpText<"Alias for --bytes">;
def : JoinedOrSeparate<["-"], "e">, Alias<encoding_EQ>, HelpText<"Alias for --encoding">;
def : JoinedOrSeparate<["-"], "t">, Alias<radix_EQ>, HelpText<"Alias for --radix">, MetaVarName<"<radix>">;
