include "llvm/Option/OptParser.td"

def sc_foo : SubCommand<"foo", "HelpText for SubCommand foo.">;

def sc_bar : SubCommand<"bar", "HelpText for SubCommand bar.",
                        "Subcommand bar <options>">;

def help : Flag<["--"], "help">, HelpText<"Subcommand <subcommand> <options>">;

def version : Flag<["-"], "version">, HelpText<"Display the version number">;

def uppercase : Flag<["-"], "uppercase", [sc_foo, sc_bar]>,
                HelpText<"Print in uppercase">;

def lowercase : Flag<["-"], "lowercase", [sc_foo]>,
                HelpText<"Print in lowercase">;
