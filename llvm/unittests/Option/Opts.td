include "llvm/Option/OptParser.td"

def OptFlag1 : OptionFlag;
def OptFlag2 : OptionFlag;
def OptFlag3 : OptionFlag;

def SubtoolVis : OptionVisibility;
def MultiLineVis: OptionVisibility;

def A : Flag<["-"], "A">, HelpText<"The A option">, Flags<[OptFlag1]>;
def AB : Flag<["-"], "AB">;
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

def Blorp : Flag<["-", "--"], "blorp">, HelpText<"The blorp option">, Flags<[OptFlag1]>;
def Blarn : Flag<["--", "-"], "blarn">, HelpText<"The blarn option">, Flags<[OptFlag1]>;
def Cramb : Joined<["/"], "cramb:">, HelpText<"The cramb option">, MetaVarName<"CRAMB">, Flags<[OptFlag1]>;
def Doopf1 : Flag<["-"], "doopf1">, HelpText<"The doopf1 option">, Flags<[OptFlag1]>;
def Doopf2 : Flag<["-"], "doopf2">, HelpText<"The doopf2 option">, Flags<[OptFlag2]>;
def Xyzzy1 : Flag<["-"], "xyzzy1">, HelpText<"The xyzzy1 option">, Visibility<[SubtoolVis]>;
def Xyzzy2 : Flag<["-"], "xyzzy2">, HelpText<"The xyzzy2 option">, Visibility<[DefaultVis]>;
def Ermgh : Joined<["--"], "ermgh">, HelpText<"The ermgh option">, MetaVarName<"ERMGH">, Flags<[OptFlag1]>;
def Fjormp : Flag<["--"], "fjormp">, HelpText<"The fjormp option">, Flags<[OptFlag1]>;

def Glorrmp_eq : Flag<["--"], "glorrmp=">;

def Blurmpq : Flag<["--"], "blurmp">;
def Blurmpq_eq : Flag<["--"], "blurmp=">;

def Q : Flag<["-"], "Q">, Visibility<[SubtoolVis]>;
def R : Flag<["-"], "R">, Visibility<[DefaultVis, SubtoolVis]>;

def multiline_help : Flag<["-"], "multiline-help">, HelpText<"This a help text that has\nmultiple lines in it">, Visibility<[MultiLineVis]>;

def multiline_help_long :
  Flag<["-"], "multiline-help-with-long-name">,
  HelpText<"This a help text that has\n"
           "multiple lines in it\n"
           "and a long name">,
  Visibility<[MultiLineVis]>;

class XOpts<string base> : KeyPathAndMacro<"X->", base> {}

def marshalled_flag_d : Flag<["-"], "marshalled-flag-d">,
  MarshallingInfoFlag<XOpts<"MarshalledFlagD">>;
def marshalled_flag_c : Flag<["-"], "marshalled-flag-c">,
  MarshallingInfoFlag<XOpts<"MarshalledFlagC">>,
  ImpliedByAnyOf<[marshalled_flag_d.KeyPath], "true">;
def marshalled_flag_b : Flag<["-"], "marshalled-flag-b">,
  MarshallingInfoFlag<XOpts<"MarshalledFlagB">>,
  ImpliedByAnyOf<[marshalled_flag_d.KeyPath], "true">;
def marshalled_flag_a : Flag<["-"], "marshalled-flag-a">,
  MarshallingInfoFlag<XOpts<"MarshalledFlagA">>,
  ImpliedByAnyOf<[marshalled_flag_c.KeyPath, marshalled_flag_b.KeyPath]>;
