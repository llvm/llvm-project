include "llvm/TableGen/Automaton.td"
include "llvm/TableGen/SearchableTable.td"

// Define a set of input token symbols.
class SymKindTy;
def SK_a : SymKindTy;
def SK_b : SymKindTy;
def SK_c : SymKindTy;
def SK_d : SymKindTy;

// Emit those as a C++ enum using SearchableTables.
def SymKind : GenericEnum {
  let FilterClass = "SymKindTy";
}

// Define a transition implementation.
class SimpleTransition<bits<2> State, SymKindTy A> : Transition {
  let NewState{1-0} = State;
  SymKindTy ActionSym = A;
}

// Token SK_a sets bit 0b01.
def : SimpleTransition<0b01, SK_a>;
// Token SK_b sets bits 0b10.
def : SimpleTransition<0b10, SK_b>;
// Token SK_c sets both bits 0b11.
def : SimpleTransition<0b11, SK_c>;

def SimpleAutomaton : GenericAutomaton {
  let TransitionClass = "SimpleTransition";
  let SymbolFields = ["ActionSym"];
  // Override the type of ActionSym from SymKindTy to the C++ type SymKind.
  string TypeOf_ActionSym = "SymKind";
}

//===----------------------------------------------------------------------===//
// TupleActionAutomaton test implementation

// Define a transition implementation.
class TupleTransition<bits<2> State, SymKindTy s1, SymKindTy s2, string s3> : Transition {
  let NewState{1-0} = State;
  SymKindTy S1 = s1;
  SymKindTy S2 = s2;
  string S3 = s3;
}

def : TupleTransition<0b01, SK_a, SK_b, "yeet">;
def : TupleTransition<0b10, SK_b, SK_b, "foo">;
def : TupleTransition<0b10, SK_c, SK_a, "foo">;

def TupleAutomaton : GenericAutomaton {
  let TransitionClass = "TupleTransition";
  let SymbolFields = ["S1", "S2", "S3"];
  string TypeOf_S1 = "SymKind";
  string TypeOf_S2 = "SymKind";
}

//===----------------------------------------------------------------------===//
// NfaAutomaton test implementation

class NfaTransition<bits<2> State, SymKindTy S> : Transition {
  let NewState{1-0} = State;
  SymKindTy A = S;
}

// Symbols a and b can transition to 0b01 or 0b11 (sets bit 0).
def : NfaTransition<0b01, SK_a>;
def : NfaTransition<0b01, SK_b>;
// Symbols a and b can also transition to 0b10 or 0b11 (sets bit 1).
def : NfaTransition<0b10, SK_a>;
def : NfaTransition<0b10, SK_b>;

def NfaAutomaton : GenericAutomaton {
  let TransitionClass = "NfaTransition";
  let SymbolFields = ["A"];
  string TypeOf_A = "SymKind";
}

//===----------------------------------------------------------------------===//
// BinPacker test implementation
//===----------------------------------------------------------------------===//
// This test generates an automaton that can pack values into bins subject to
// constraints. There are 6 possible bins, and the input tokens are constraint
// types. Some input types span two bins.

// The symbol type for a bin constraint. We use lists of ints as a tblgen hack
// to conditionally generate defs within multiclasses based on record
// information. A bin is nonempty (has a dummy one-element value) if enabled.
class BinRequirementKind {
  list<int> Bin0 = [];
  list<int> Bin1 = [];
  list<int> Bin2 = [];
  list<int> Bin3 = [];
  list<int> Bin4 = [];
  list<int> Bin5 = [];
}
// Can use bins {0-3}
def BRK_0_to_4    : BinRequirementKind { let Bin0 = [1]; let Bin1 = [1]; let Bin2 = [1]; let Bin3 = [1]; }
// Can use bins {0-3} but only evens (0 and 2).
def BRK_0_to_4_lo : BinRequirementKind { let Bin0 = [1]; let Bin2 = [1]; }
// Can use bins {0-3} but only odds (1 and 3).
def BRK_0_to_4_hi : BinRequirementKind { let Bin1 = [1]; let Bin3 = [1]; }
// Can use bins {0-3} but only even-odd pairs (0+1 or 1+2).
def BRK_0_to_4_dbl : BinRequirementKind { let Bin0 = [1]; let Bin2 = [1]; }
def BRK_0_to_6 :    BinRequirementKind { let Bin0 = [1]; let Bin1 = [1]; let Bin2 = [1];
                                         let Bin3 = [1]; let Bin4 = [1]; let Bin5 = [1]; }
def BRK_0_to_6_lo : BinRequirementKind { let Bin0 = [1]; let Bin2 = [1]; let Bin4 = [1]; }
def BRK_0_to_6_hi : BinRequirementKind { let Bin1 = [1]; let Bin3 = [1]; let Bin5 = [1]; }
def BRK_0_to_6_dbl : BinRequirementKind { let Bin0 = [1]; let Bin2 = [1]; let Bin4 = [1]; }
def BRK_2_to_6 :    BinRequirementKind { let Bin2 = [1];
                                         let Bin3 = [1]; let Bin4 = [1]; let Bin5 = [1]; }
def BRK_2_to_6_lo : BinRequirementKind { let Bin2 = [1]; let Bin4 = [1]; }
def BRK_2_to_6_hi : BinRequirementKind { let Bin3 = [1]; let Bin5 = [1];}
def BRK_2_to_6_dbl : BinRequirementKind { let Bin2 = [1]; let Bin4 = [1]; }
def BRK_2_to_4 :    BinRequirementKind { let Bin2 = [1]; let Bin3 = [1]; }
def BRK_2_to_4_lo : BinRequirementKind { let Bin2 = [1]; }
def BRK_2_to_4_hi : BinRequirementKind { let Bin3 = [1]; }
def BRK_2_to_4_dbl : BinRequirementKind { let Bin2 = [1]; }

def BinRequirementKindEnum : GenericEnum {
  let FilterClass = "BinRequirementKind";
}

// The transition class is trivial; it just contains the constraint symbol.
class BinTransition : Transition {
  BinRequirementKind Sym;
}

// Mixin that occupies a single bin.
class Bin0 : BinTransition { let NewState{0} = 1; }
class Bin1 : BinTransition { let NewState{1} = 1; }
class Bin2 : BinTransition { let NewState{2} = 1;}
class Bin3 : BinTransition { let NewState{3} = 1; }
class Bin4 : BinTransition { let NewState{4} = 1;}
class Bin5 : BinTransition { let NewState{5} = 1; }
// Mixin that occupies a pair of bins (even-odd pairs).
class Bin01 : BinTransition { let NewState{0,1} = 0b11; }
class Bin23 : BinTransition { let NewState{2,3} = 0b11; }
class Bin45 : BinTransition { let NewState{4,5} = 0b11; }

// Instantiate all possible bin assignments for E.
multiclass BinAssignments<BinRequirementKind E> {
  let Sym = E in {
    // Note the tablegen hack to conditionally instantiate a def based on E.
    foreach x = E.Bin0 in { def : Bin0; }
    foreach x = E.Bin1 in { def : Bin1; }
    foreach x = E.Bin2 in { def : Bin2; }
    foreach x = E.Bin3 in { def : Bin3; }
    foreach x = E.Bin4 in { def : Bin4; }
    foreach x = E.Bin5 in { def : Bin5; }
  }
}

// Instantiate all possible bin assignments for E, which spans even-odd pairs.
multiclass DblBinAssignments<BinRequirementKind E> {
  let Sym = E in {
    foreach x = E.Bin0 in { def : Bin01; }
    foreach x = E.Bin2 in { def : Bin23; }
    foreach x = E.Bin4 in { def : Bin45; }
  }
}

defm : BinAssignments<BRK_0_to_4>;
defm : DblBinAssignments<BRK_0_to_4_dbl>;
defm : BinAssignments<BRK_0_to_4_lo>;
defm : BinAssignments<BRK_0_to_4_hi>;
defm : BinAssignments<BRK_0_to_6>;
defm : DblBinAssignments<BRK_0_to_6_dbl>;
defm : BinAssignments<BRK_0_to_6_lo>;
defm : BinAssignments<BRK_0_to_6_hi>;
defm : BinAssignments<BRK_2_to_6>;
defm : DblBinAssignments<BRK_2_to_6_dbl>;
defm : BinAssignments<BRK_2_to_6_lo>;
defm : BinAssignments<BRK_2_to_6_hi>;
defm : BinAssignments<BRK_2_to_4>;
defm : DblBinAssignments<BRK_2_to_4_dbl>;
defm : BinAssignments<BRK_2_to_4_lo>;
defm : BinAssignments<BRK_2_to_4_hi>;

def BinPackerAutomaton : GenericAutomaton {
  let TransitionClass = "BinTransition";
  let SymbolFields = ["Sym"];
  string TypeOf_Sym = "BinRequirementKindEnum";
}


