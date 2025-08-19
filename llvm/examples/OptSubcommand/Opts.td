include "llvm/Option/OptParser.td"

def sc_foo : Subcommand<"foo", "HelpText for Subcommand foo.">;

def sc_bar : Subcommand<"bar", "HelpText for Subcommand bar.">;

def help : Flag<["--"], "help">, HelpText<"Top Level Help Text for the tool.">;

def version : Flag<["-"], "version">,
              HelpText<"Toplevel Display the version number">;

def uppercase : Flag<["-"], "uppercase", [sc_foo, sc_bar]>,
                HelpText<"Print in uppercase">;

def lowercase : Flag<["-"], "lowercase", [sc_foo]>,
                HelpText<"Print in lowercase">;
