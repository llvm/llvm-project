include "llvm/Option/OptParser.td"

def OptFlag1 : OptionFlag;
def OptFlag2 : OptionFlag;
def OptFlag3 : OptionFlag;

def A : Flag<["-"], "A">, HelpText<"The A option">, Flags<[OptFlag1]>;
def B : Joined<["-"], "B">, HelpText<"The B option">, MetaVarName<"B">, Flags<[OptFlag2]>;
def C : Separate<["-"], "C">, HelpText<"The C option">, MetaVarName<"C">, Flags<[OptFlag1]>;
def SLASH_C : Separate<["/", "-"], "C">, HelpText<"The C option">, MetaVarName<"C">, Flags<[OptFlag3]>;
def D : CommaJoined<["-"], "D">, HelpText<"The D option">, MetaVarName<"D">;
def E : MultiArg<["-"], "E", 2>, Flags<[OptFlag1, OptFlag2]>;
def F : JoinedOrSeparate<["-"], "F">, HelpText<"The F option">, MetaVarName<"F">;
def G : JoinedAndSeparate<["-"], "G">, HelpText<"The G option">, MetaVarName<"G">;

def Ceq : Joined<["-", "--"], "C=">, Alias<C>, Flags<[OptFlag1]>;

def H : Flag<["-"], "H">, Flags<[HelpHidden]>;

def my_group : OptionGroup<"my group">;
def I : Flag<["-"], "I">, Alias<H>, Group<my_group>;

def J : Flag<["-"], "J">, Alias<B>, AliasArgs<["foo"]>;
def Joo : Flag<["-"], "Joo">, Alias<B>, AliasArgs<["bar"]>;

def K : Flag<["-"], "K">, Alias<B>;

def Slurp : Option<["-"], "slurp", KIND_REMAINING_ARGS>;

def SlurpJoined : Option<["-"], "slurpjoined", KIND_REMAINING_ARGS_JOINED>;
