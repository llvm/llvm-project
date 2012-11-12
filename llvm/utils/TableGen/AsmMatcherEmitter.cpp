//===- AsmMatcherEmitter.cpp - Generate an assembly matcher ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This tablegen backend emits a target specifier matcher for converting parsed
// assembly operands in the MCInst structures. It also emits a matcher for
// custom operand parsing.
//
// Converting assembly operands into MCInst structures
// ---------------------------------------------------
//
// The input to the target specific matcher is a list of literal tokens and
// operands. The target specific parser should generally eliminate any syntax
// which is not relevant for matching; for example, comma tokens should have
// already been consumed and eliminated by the parser. Most instructions will
// end up with a single literal token (the instruction name) and some number of
// operands.
//
// Some example inputs, for X86:
//   'addl' (immediate ...) (register ...)
//   'add' (immediate ...) (memory ...)
//   'call' '*' %epc
//
// The assembly matcher is responsible for converting this input into a precise
// machine instruction (i.e., an instruction with a well defined encoding). This
// mapping has several properties which complicate matching:
//
//  - It may be ambiguous; many architectures can legally encode particular
//    variants of an instruction in different ways (for example, using a smaller
//    encoding for small immediates). Such ambiguities should never be
//    arbitrarily resolved by the assembler, the assembler is always responsible
//    for choosing the "best" available instruction.
//
//  - It may depend on the subtarget or the assembler context. Instructions
//    which are invalid for the current mode, but otherwise unambiguous (e.g.,
//    an SSE instruction in a file being assembled for i486) should be accepted
//    and rejected by the assembler front end. However, if the proper encoding
//    for an instruction is dependent on the assembler context then the matcher
//    is responsible for selecting the correct machine instruction for the
//    current mode.
//
// The core matching algorithm attempts to exploit the regularity in most
// instruction sets to quickly determine the set of possibly matching
// instructions, and the simplify the generated code. Additionally, this helps
// to ensure that the ambiguities are intentionally resolved by the user.
//
// The matching is divided into two distinct phases:
//
//   1. Classification: Each operand is mapped to the unique set which (a)
//      contains it, and (b) is the largest such subset for which a single
//      instruction could match all members.
//
//      For register classes, we can generate these subgroups automatically. For
//      arbitrary operands, we expect the user to define the classes and their
//      relations to one another (for example, 8-bit signed immediates as a
//      subset of 32-bit immediates).
//
//      By partitioning the operands in this way, we guarantee that for any
//      tuple of classes, any single instruction must match either all or none
//      of the sets of operands which could classify to that tuple.
//
//      In addition, the subset relation amongst classes induces a partial order
//      on such tuples, which we use to resolve ambiguities.
//
//   2. The input can now be treated as a tuple of classes (static tokens are
//      simple singleton sets). Each such tuple should generally map to a single
//      instruction (we currently ignore cases where this isn't true, whee!!!),
//      which we can emit a simple matcher for.
//
// Custom Operand Parsing
// ----------------------
//
//  Some targets need a custom way to parse operands, some specific instructions
//  can contain arguments that can represent processor flags and other kinds of
//  identifiers that need to be mapped to specific values in the final encoded
//  instructions. The target specific custom operand parsing works in the
//  following way:
//
//   1. A operand match table is built, each entry contains a mnemonic, an
//      operand class, a mask for all operand positions for that same
//      class/mnemonic and target features to be checked while trying to match.
//
//   2. The operand matcher will try every possible entry with the same
//      mnemonic and will check if the target feature for this mnemonic also
//      matches. After that, if the operand to be matched has its index
//      present in the mask, a successful match occurs. Otherwise, fallback
//      to the regular operand parsing.
//
//   3. For a match success, each operand class that has a 'ParserMethod'
//      becomes part of a switch from where the custom method is called.
//
//===----------------------------------------------------------------------===//

#include "CodeGenTarget.h"
#include "StringToOffsetTable.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/StringMatcher.h"
#include "llvm/TableGen/TableGenBackend.h"
#include <cassert>
#include <map>
#include <set>
using namespace llvm;

static cl::opt<std::string>
MatchPrefix("match-prefix", cl::init(""),
            cl::desc("Only match instructions with the given prefix"));

namespace {
class AsmMatcherInfo;
struct SubtargetFeatureInfo;

class AsmMatcherEmitter {
  RecordKeeper &Records;
public:
  AsmMatcherEmitter(RecordKeeper &R) : Records(R) {}

  void run(raw_ostream &o);
};

/// ClassInfo - Helper class for storing the information about a particular
/// class of operands which can be matched.
struct ClassInfo {
  enum ClassInfoKind {
    /// Invalid kind, for use as a sentinel value.
    Invalid = 0,

    /// The class for a particular token.
    Token,

    /// The (first) register class, subsequent register classes are
    /// RegisterClass0+1, and so on.
    RegisterClass0,

    /// The (first) user defined class, subsequent user defined classes are
    /// UserClass0+1, and so on.
    UserClass0 = 1<<16
  };

  /// Kind - The class kind, which is either a predefined kind, or (UserClass0 +
  /// N) for the Nth user defined class.
  unsigned Kind;

  /// SuperClasses - The super classes of this class. Note that for simplicities
  /// sake user operands only record their immediate super class, while register
  /// operands include all superclasses.
  std::vector<ClassInfo*> SuperClasses;

  /// Name - The full class name, suitable for use in an enum.
  std::string Name;

  /// ClassName - The unadorned generic name for this class (e.g., Token).
  std::string ClassName;

  /// ValueName - The name of the value this class represents; for a token this
  /// is the literal token string, for an operand it is the TableGen class (or
  /// empty if this is a derived class).
  std::string ValueName;

  /// PredicateMethod - The name of the operand method to test whether the
  /// operand matches this class; this is not valid for Token or register kinds.
  std::string PredicateMethod;

  /// RenderMethod - The name of the operand method to add this operand to an
  /// MCInst; this is not valid for Token or register kinds.
  std::string RenderMethod;

  /// ParserMethod - The name of the operand method to do a target specific
  /// parsing on the operand.
  std::string ParserMethod;

  /// For register classes, the records for all the registers in this class.
  std::set<Record*> Registers;

  /// For custom match classes, he diagnostic kind for when the predicate fails.
  std::string DiagnosticType;
public:
  /// isRegisterClass() - Check if this is a register class.
  bool isRegisterClass() const {
    return Kind >= RegisterClass0 && Kind < UserClass0;
  }

  /// isUserClass() - Check if this is a user defined class.
  bool isUserClass() const {
    return Kind >= UserClass0;
  }

  /// isRelatedTo - Check whether this class is "related" to \p RHS. Classes
  /// are related if they are in the same class hierarchy.
  bool isRelatedTo(const ClassInfo &RHS) const {
    // Tokens are only related to tokens.
    if (Kind == Token || RHS.Kind == Token)
      return Kind == Token && RHS.Kind == Token;

    // Registers classes are only related to registers classes, and only if
    // their intersection is non-empty.
    if (isRegisterClass() || RHS.isRegisterClass()) {
      if (!isRegisterClass() || !RHS.isRegisterClass())
        return false;

      std::set<Record*> Tmp;
      std::insert_iterator< std::set<Record*> > II(Tmp, Tmp.begin());
      std::set_intersection(Registers.begin(), Registers.end(),
                            RHS.Registers.begin(), RHS.Registers.end(),
                            II);

      return !Tmp.empty();
    }

    // Otherwise we have two users operands; they are related if they are in the
    // same class hierarchy.
    //
    // FIXME: This is an oversimplification, they should only be related if they
    // intersect, however we don't have that information.
    assert(isUserClass() && RHS.isUserClass() && "Unexpected class!");
    const ClassInfo *Root = this;
    while (!Root->SuperClasses.empty())
      Root = Root->SuperClasses.front();

    const ClassInfo *RHSRoot = &RHS;
    while (!RHSRoot->SuperClasses.empty())
      RHSRoot = RHSRoot->SuperClasses.front();

    return Root == RHSRoot;
  }

  /// isSubsetOf - Test whether this class is a subset of \p RHS.
  bool isSubsetOf(const ClassInfo &RHS) const {
    // This is a subset of RHS if it is the same class...
    if (this == &RHS)
      return true;

    // ... or if any of its super classes are a subset of RHS.
    for (std::vector<ClassInfo*>::const_iterator it = SuperClasses.begin(),
           ie = SuperClasses.end(); it != ie; ++it)
      if ((*it)->isSubsetOf(RHS))
        return true;

    return false;
  }

  /// operator< - Compare two classes.
  bool operator<(const ClassInfo &RHS) const {
    if (this == &RHS)
      return false;

    // Unrelated classes can be ordered by kind.
    if (!isRelatedTo(RHS))
      return Kind < RHS.Kind;

    switch (Kind) {
    case Invalid:
      llvm_unreachable("Invalid kind!");

    default:
      // This class precedes the RHS if it is a proper subset of the RHS.
      if (isSubsetOf(RHS))
        return true;
      if (RHS.isSubsetOf(*this))
        return false;

      // Otherwise, order by name to ensure we have a total ordering.
      return ValueName < RHS.ValueName;
    }
  }
};

namespace {
/// Sort ClassInfo pointers independently of pointer value.
struct LessClassInfoPtr {
  bool operator()(const ClassInfo *LHS, const ClassInfo *RHS) const {
    return *LHS < *RHS;
  }
};
}

/// MatchableInfo - Helper class for storing the necessary information for an
/// instruction or alias which is capable of being matched.
struct MatchableInfo {
  struct AsmOperand {
    /// Token - This is the token that the operand came from.
    StringRef Token;

    /// The unique class instance this operand should match.
    ClassInfo *Class;

    /// The operand name this is, if anything.
    StringRef SrcOpName;

    /// The suboperand index within SrcOpName, or -1 for the entire operand.
    int SubOpIdx;

    /// Register record if this token is singleton register.
    Record *SingletonReg;

    explicit AsmOperand(StringRef T) : Token(T), Class(0), SubOpIdx(-1),
                                       SingletonReg(0) {}
  };

  /// ResOperand - This represents a single operand in the result instruction
  /// generated by the match.  In cases (like addressing modes) where a single
  /// assembler operand expands to multiple MCOperands, this represents the
  /// single assembler operand, not the MCOperand.
  struct ResOperand {
    enum {
      /// RenderAsmOperand - This represents an operand result that is
      /// generated by calling the render method on the assembly operand.  The
      /// corresponding AsmOperand is specified by AsmOperandNum.
      RenderAsmOperand,

      /// TiedOperand - This represents a result operand that is a duplicate of
      /// a previous result operand.
      TiedOperand,

      /// ImmOperand - This represents an immediate value that is dumped into
      /// the operand.
      ImmOperand,

      /// RegOperand - This represents a fixed register that is dumped in.
      RegOperand
    } Kind;

    union {
      /// This is the operand # in the AsmOperands list that this should be
      /// copied from.
      unsigned AsmOperandNum;

      /// TiedOperandNum - This is the (earlier) result operand that should be
      /// copied from.
      unsigned TiedOperandNum;

      /// ImmVal - This is the immediate value added to the instruction.
      int64_t ImmVal;

      /// Register - This is the register record.
      Record *Register;
    };

    /// MINumOperands - The number of MCInst operands populated by this
    /// operand.
    unsigned MINumOperands;

    static ResOperand getRenderedOp(unsigned AsmOpNum, unsigned NumOperands) {
      ResOperand X;
      X.Kind = RenderAsmOperand;
      X.AsmOperandNum = AsmOpNum;
      X.MINumOperands = NumOperands;
      return X;
    }

    static ResOperand getTiedOp(unsigned TiedOperandNum) {
      ResOperand X;
      X.Kind = TiedOperand;
      X.TiedOperandNum = TiedOperandNum;
      X.MINumOperands = 1;
      return X;
    }

    static ResOperand getImmOp(int64_t Val) {
      ResOperand X;
      X.Kind = ImmOperand;
      X.ImmVal = Val;
      X.MINumOperands = 1;
      return X;
    }

    static ResOperand getRegOp(Record *Reg) {
      ResOperand X;
      X.Kind = RegOperand;
      X.Register = Reg;
      X.MINumOperands = 1;
      return X;
    }
  };

  /// AsmVariantID - Target's assembly syntax variant no.
  int AsmVariantID;

  /// TheDef - This is the definition of the instruction or InstAlias that this
  /// matchable came from.
  Record *const TheDef;

  /// DefRec - This is the definition that it came from.
  PointerUnion<const CodeGenInstruction*, const CodeGenInstAlias*> DefRec;

  const CodeGenInstruction *getResultInst() const {
    if (DefRec.is<const CodeGenInstruction*>())
      return DefRec.get<const CodeGenInstruction*>();
    return DefRec.get<const CodeGenInstAlias*>()->ResultInst;
  }

  /// ResOperands - This is the operand list that should be built for the result
  /// MCInst.
  SmallVector<ResOperand, 8> ResOperands;

  /// AsmString - The assembly string for this instruction (with variants
  /// removed), e.g. "movsx $src, $dst".
  std::string AsmString;

  /// Mnemonic - This is the first token of the matched instruction, its
  /// mnemonic.
  StringRef Mnemonic;

  /// AsmOperands - The textual operands that this instruction matches,
  /// annotated with a class and where in the OperandList they were defined.
  /// This directly corresponds to the tokenized AsmString after the mnemonic is
  /// removed.
  SmallVector<AsmOperand, 8> AsmOperands;

  /// Predicates - The required subtarget features to match this instruction.
  SmallVector<SubtargetFeatureInfo*, 4> RequiredFeatures;

  /// ConversionFnKind - The enum value which is passed to the generated
  /// convertToMCInst to convert parsed operands into an MCInst for this
  /// function.
  std::string ConversionFnKind;

  MatchableInfo(const CodeGenInstruction &CGI)
    : AsmVariantID(0), TheDef(CGI.TheDef), DefRec(&CGI),
      AsmString(CGI.AsmString) {
  }

  MatchableInfo(const CodeGenInstAlias *Alias)
    : AsmVariantID(0), TheDef(Alias->TheDef), DefRec(Alias),
      AsmString(Alias->AsmString) {
  }

  // Two-operand aliases clone from the main matchable, but mark the second
  // operand as a tied operand of the first for purposes of the assembler.
  void formTwoOperandAlias(StringRef Constraint);

  void initialize(const AsmMatcherInfo &Info,
                  SmallPtrSet<Record*, 16> &SingletonRegisters,
                  int AsmVariantNo, std::string &RegisterPrefix);

  /// validate - Return true if this matchable is a valid thing to match against
  /// and perform a bunch of validity checking.
  bool validate(StringRef CommentDelimiter, bool Hack) const;

  /// extractSingletonRegisterForAsmOperand - Extract singleton register,
  /// if present, from specified token.
  void
  extractSingletonRegisterForAsmOperand(unsigned i, const AsmMatcherInfo &Info,
                                        std::string &RegisterPrefix);

  /// findAsmOperand - Find the AsmOperand with the specified name and
  /// suboperand index.
  int findAsmOperand(StringRef N, int SubOpIdx) const {
    for (unsigned i = 0, e = AsmOperands.size(); i != e; ++i)
      if (N == AsmOperands[i].SrcOpName &&
          SubOpIdx == AsmOperands[i].SubOpIdx)
        return i;
    return -1;
  }

  /// findAsmOperandNamed - Find the first AsmOperand with the specified name.
  /// This does not check the suboperand index.
  int findAsmOperandNamed(StringRef N) const {
    for (unsigned i = 0, e = AsmOperands.size(); i != e; ++i)
      if (N == AsmOperands[i].SrcOpName)
        return i;
    return -1;
  }

  void buildInstructionResultOperands();
  void buildAliasResultOperands();

  /// operator< - Compare two matchables.
  bool operator<(const MatchableInfo &RHS) const {
    // The primary comparator is the instruction mnemonic.
    if (Mnemonic != RHS.Mnemonic)
      return Mnemonic < RHS.Mnemonic;

    if (AsmOperands.size() != RHS.AsmOperands.size())
      return AsmOperands.size() < RHS.AsmOperands.size();

    // Compare lexicographically by operand. The matcher validates that other
    // orderings wouldn't be ambiguous using \see couldMatchAmbiguouslyWith().
    for (unsigned i = 0, e = AsmOperands.size(); i != e; ++i) {
      if (*AsmOperands[i].Class < *RHS.AsmOperands[i].Class)
        return true;
      if (*RHS.AsmOperands[i].Class < *AsmOperands[i].Class)
        return false;
    }

    // Give matches that require more features higher precedence. This is useful
    // because we cannot define AssemblerPredicates with the negation of
    // processor features. For example, ARM v6 "nop" may be either a HINT or
    // MOV. With v6, we want to match HINT. The assembler has no way to
    // predicate MOV under "NoV6", but HINT will always match first because it
    // requires V6 while MOV does not.
    if (RequiredFeatures.size() != RHS.RequiredFeatures.size())
      return RequiredFeatures.size() > RHS.RequiredFeatures.size();

    return false;
  }

  /// couldMatchAmbiguouslyWith - Check whether this matchable could
  /// ambiguously match the same set of operands as \p RHS (without being a
  /// strictly superior match).
  bool couldMatchAmbiguouslyWith(const MatchableInfo &RHS) {
    // The primary comparator is the instruction mnemonic.
    if (Mnemonic != RHS.Mnemonic)
      return false;

    // The number of operands is unambiguous.
    if (AsmOperands.size() != RHS.AsmOperands.size())
      return false;

    // Otherwise, make sure the ordering of the two instructions is unambiguous
    // by checking that either (a) a token or operand kind discriminates them,
    // or (b) the ordering among equivalent kinds is consistent.

    // Tokens and operand kinds are unambiguous (assuming a correct target
    // specific parser).
    for (unsigned i = 0, e = AsmOperands.size(); i != e; ++i)
      if (AsmOperands[i].Class->Kind != RHS.AsmOperands[i].Class->Kind ||
          AsmOperands[i].Class->Kind == ClassInfo::Token)
        if (*AsmOperands[i].Class < *RHS.AsmOperands[i].Class ||
            *RHS.AsmOperands[i].Class < *AsmOperands[i].Class)
          return false;

    // Otherwise, this operand could commute if all operands are equivalent, or
    // there is a pair of operands that compare less than and a pair that
    // compare greater than.
    bool HasLT = false, HasGT = false;
    for (unsigned i = 0, e = AsmOperands.size(); i != e; ++i) {
      if (*AsmOperands[i].Class < *RHS.AsmOperands[i].Class)
        HasLT = true;
      if (*RHS.AsmOperands[i].Class < *AsmOperands[i].Class)
        HasGT = true;
    }

    return !(HasLT ^ HasGT);
  }

  void dump();

private:
  void tokenizeAsmString(const AsmMatcherInfo &Info);
};

/// SubtargetFeatureInfo - Helper class for storing information on a subtarget
/// feature which participates in instruction matching.
struct SubtargetFeatureInfo {
  /// \brief The predicate record for this feature.
  Record *TheDef;

  /// \brief An unique index assigned to represent this feature.
  unsigned Index;

  SubtargetFeatureInfo(Record *D, unsigned Idx) : TheDef(D), Index(Idx) {}

  /// \brief The name of the enumerated constant identifying this feature.
  std::string getEnumName() const {
    return "Feature_" + TheDef->getName();
  }
};

struct OperandMatchEntry {
  unsigned OperandMask;
  MatchableInfo* MI;
  ClassInfo *CI;

  static OperandMatchEntry create(MatchableInfo* mi, ClassInfo *ci,
                                  unsigned opMask) {
    OperandMatchEntry X;
    X.OperandMask = opMask;
    X.CI = ci;
    X.MI = mi;
    return X;
  }
};


class AsmMatcherInfo {
public:
  /// Tracked Records
  RecordKeeper &Records;

  /// The tablegen AsmParser record.
  Record *AsmParser;

  /// Target - The target information.
  CodeGenTarget &Target;

  /// The classes which are needed for matching.
  std::vector<ClassInfo*> Classes;

  /// The information on the matchables to match.
  std::vector<MatchableInfo*> Matchables;

  /// Info for custom matching operands by user defined methods.
  std::vector<OperandMatchEntry> OperandMatchInfo;

  /// Map of Register records to their class information.
  typedef std::map<Record*, ClassInfo*, LessRecordByID> RegisterClassesTy;
  RegisterClassesTy RegisterClasses;

  /// Map of Predicate records to their subtarget information.
  std::map<Record*, SubtargetFeatureInfo*> SubtargetFeatures;

  /// Map of AsmOperandClass records to their class information.
  std::map<Record*, ClassInfo*> AsmOperandClasses;

private:
  /// Map of token to class information which has already been constructed.
  std::map<std::string, ClassInfo*> TokenClasses;

  /// Map of RegisterClass records to their class information.
  std::map<Record*, ClassInfo*> RegisterClassClasses;

private:
  /// getTokenClass - Lookup or create the class for the given token.
  ClassInfo *getTokenClass(StringRef Token);

  /// getOperandClass - Lookup or create the class for the given operand.
  ClassInfo *getOperandClass(const CGIOperandList::OperandInfo &OI,
                             int SubOpIdx);
  ClassInfo *getOperandClass(Record *Rec, int SubOpIdx);

  /// buildRegisterClasses - Build the ClassInfo* instances for register
  /// classes.
  void buildRegisterClasses(SmallPtrSet<Record*, 16> &SingletonRegisters);

  /// buildOperandClasses - Build the ClassInfo* instances for user defined
  /// operand classes.
  void buildOperandClasses();

  void buildInstructionOperandReference(MatchableInfo *II, StringRef OpName,
                                        unsigned AsmOpIdx);
  void buildAliasOperandReference(MatchableInfo *II, StringRef OpName,
                                  MatchableInfo::AsmOperand &Op);

public:
  AsmMatcherInfo(Record *AsmParser,
                 CodeGenTarget &Target,
                 RecordKeeper &Records);

  /// buildInfo - Construct the various tables used during matching.
  void buildInfo();

  /// buildOperandMatchInfo - Build the necessary information to handle user
  /// defined operand parsing methods.
  void buildOperandMatchInfo();

  /// getSubtargetFeature - Lookup or create the subtarget feature info for the
  /// given operand.
  SubtargetFeatureInfo *getSubtargetFeature(Record *Def) const {
    assert(Def->isSubClassOf("Predicate") && "Invalid predicate type!");
    std::map<Record*, SubtargetFeatureInfo*>::const_iterator I =
      SubtargetFeatures.find(Def);
    return I == SubtargetFeatures.end() ? 0 : I->second;
  }

  RecordKeeper &getRecords() const {
    return Records;
  }
};

} // End anonymous namespace

void MatchableInfo::dump() {
  errs() << TheDef->getName() << " -- " << "flattened:\"" << AsmString <<"\"\n";

  for (unsigned i = 0, e = AsmOperands.size(); i != e; ++i) {
    AsmOperand &Op = AsmOperands[i];
    errs() << "  op[" << i << "] = " << Op.Class->ClassName << " - ";
    errs() << '\"' << Op.Token << "\"\n";
  }
}

static std::pair<StringRef, StringRef>
parseTwoOperandConstraint(StringRef S, ArrayRef<SMLoc> Loc) {
  // Split via the '='.
  std::pair<StringRef, StringRef> Ops = S.split('=');
  if (Ops.second == "")
    PrintFatalError(Loc, "missing '=' in two-operand alias constraint");
  // Trim whitespace and the leading '$' on the operand names.
  size_t start = Ops.first.find_first_of('$');
  if (start == std::string::npos)
    PrintFatalError(Loc, "expected '$' prefix on asm operand name");
  Ops.first = Ops.first.slice(start + 1, std::string::npos);
  size_t end = Ops.first.find_last_of(" \t");
  Ops.first = Ops.first.slice(0, end);
  // Now the second operand.
  start = Ops.second.find_first_of('$');
  if (start == std::string::npos)
    PrintFatalError(Loc, "expected '$' prefix on asm operand name");
  Ops.second = Ops.second.slice(start + 1, std::string::npos);
  end = Ops.second.find_last_of(" \t");
  Ops.first = Ops.first.slice(0, end);
  return Ops;
}

void MatchableInfo::formTwoOperandAlias(StringRef Constraint) {
  // Figure out which operands are aliased and mark them as tied.
  std::pair<StringRef, StringRef> Ops =
    parseTwoOperandConstraint(Constraint, TheDef->getLoc());

  // Find the AsmOperands that refer to the operands we're aliasing.
  int SrcAsmOperand = findAsmOperandNamed(Ops.first);
  int DstAsmOperand = findAsmOperandNamed(Ops.second);
  if (SrcAsmOperand == -1)
    PrintFatalError(TheDef->getLoc(),
                  "unknown source two-operand alias operand '" +
                  Ops.first.str() + "'.");
  if (DstAsmOperand == -1)
    PrintFatalError(TheDef->getLoc(),
                  "unknown destination two-operand alias operand '" +
                  Ops.second.str() + "'.");

  // Find the ResOperand that refers to the operand we're aliasing away
  // and update it to refer to the combined operand instead.
  for (unsigned i = 0, e = ResOperands.size(); i != e; ++i) {
    ResOperand &Op = ResOperands[i];
    if (Op.Kind == ResOperand::RenderAsmOperand &&
        Op.AsmOperandNum == (unsigned)SrcAsmOperand) {
      Op.AsmOperandNum = DstAsmOperand;
      break;
    }
  }
  // Remove the AsmOperand for the alias operand.
  AsmOperands.erase(AsmOperands.begin() + SrcAsmOperand);
  // Adjust the ResOperand references to any AsmOperands that followed
  // the one we just deleted.
  for (unsigned i = 0, e = ResOperands.size(); i != e; ++i) {
    ResOperand &Op = ResOperands[i];
    switch(Op.Kind) {
    default:
      // Nothing to do for operands that don't reference AsmOperands.
      break;
    case ResOperand::RenderAsmOperand:
      if (Op.AsmOperandNum > (unsigned)SrcAsmOperand)
        --Op.AsmOperandNum;
      break;
    case ResOperand::TiedOperand:
      if (Op.TiedOperandNum > (unsigned)SrcAsmOperand)
        --Op.TiedOperandNum;
      break;
    }
  }
}

void MatchableInfo::initialize(const AsmMatcherInfo &Info,
                               SmallPtrSet<Record*, 16> &SingletonRegisters,
                               int AsmVariantNo, std::string &RegisterPrefix) {
  AsmVariantID = AsmVariantNo;
  AsmString =
    CodeGenInstruction::FlattenAsmStringVariants(AsmString, AsmVariantNo);

  tokenizeAsmString(Info);

  // Compute the require features.
  std::vector<Record*> Predicates =TheDef->getValueAsListOfDefs("Predicates");
  for (unsigned i = 0, e = Predicates.size(); i != e; ++i)
    if (SubtargetFeatureInfo *Feature =
        Info.getSubtargetFeature(Predicates[i]))
      RequiredFeatures.push_back(Feature);

  // Collect singleton registers, if used.
  for (unsigned i = 0, e = AsmOperands.size(); i != e; ++i) {
    extractSingletonRegisterForAsmOperand(i, Info, RegisterPrefix);
    if (Record *Reg = AsmOperands[i].SingletonReg)
      SingletonRegisters.insert(Reg);
  }
}

/// tokenizeAsmString - Tokenize a simplified assembly string.
void MatchableInfo::tokenizeAsmString(const AsmMatcherInfo &Info) {
  StringRef String = AsmString;
  unsigned Prev = 0;
  bool InTok = true;
  for (unsigned i = 0, e = String.size(); i != e; ++i) {
    switch (String[i]) {
    case '[':
    case ']':
    case '*':
    case '!':
    case ' ':
    case '\t':
    case ',':
      if (InTok) {
        AsmOperands.push_back(AsmOperand(String.slice(Prev, i)));
        InTok = false;
      }
      if (!isspace(String[i]) && String[i] != ',')
        AsmOperands.push_back(AsmOperand(String.substr(i, 1)));
      Prev = i + 1;
      break;

    case '\\':
      if (InTok) {
        AsmOperands.push_back(AsmOperand(String.slice(Prev, i)));
        InTok = false;
      }
      ++i;
      assert(i != String.size() && "Invalid quoted character");
      AsmOperands.push_back(AsmOperand(String.substr(i, 1)));
      Prev = i + 1;
      break;

    case '$': {
      if (InTok) {
        AsmOperands.push_back(AsmOperand(String.slice(Prev, i)));
        InTok = false;
      }

      // If this isn't "${", treat like a normal token.
      if (i + 1 == String.size() || String[i + 1] != '{') {
        Prev = i;
        break;
      }

      StringRef::iterator End = std::find(String.begin() + i, String.end(),'}');
      assert(End != String.end() && "Missing brace in operand reference!");
      size_t EndPos = End - String.begin();
      AsmOperands.push_back(AsmOperand(String.slice(i, EndPos+1)));
      Prev = EndPos + 1;
      i = EndPos;
      break;
    }

    case '.':
      if (InTok)
        AsmOperands.push_back(AsmOperand(String.slice(Prev, i)));
      Prev = i;
      InTok = true;
      break;

    default:
      InTok = true;
    }
  }
  if (InTok && Prev != String.size())
    AsmOperands.push_back(AsmOperand(String.substr(Prev)));

  // The first token of the instruction is the mnemonic, which must be a
  // simple string, not a $foo variable or a singleton register.
  if (AsmOperands.empty())
    PrintFatalError(TheDef->getLoc(),
                  "Instruction '" + TheDef->getName() + "' has no tokens");
  Mnemonic = AsmOperands[0].Token;
  if (Mnemonic.empty())
    PrintFatalError(TheDef->getLoc(),
                  "Missing instruction mnemonic");
  // FIXME : Check and raise an error if it is a register.
  if (Mnemonic[0] == '$')
    PrintFatalError(TheDef->getLoc(),
                  "Invalid instruction mnemonic '" + Mnemonic.str() + "'!");

  // Remove the first operand, it is tracked in the mnemonic field.
  AsmOperands.erase(AsmOperands.begin());
}

bool MatchableInfo::validate(StringRef CommentDelimiter, bool Hack) const {
  // Reject matchables with no .s string.
  if (AsmString.empty())
    PrintFatalError(TheDef->getLoc(), "instruction with empty asm string");

  // Reject any matchables with a newline in them, they should be marked
  // isCodeGenOnly if they are pseudo instructions.
  if (AsmString.find('\n') != std::string::npos)
    PrintFatalError(TheDef->getLoc(),
                  "multiline instruction is not valid for the asmparser, "
                  "mark it isCodeGenOnly");

  // Remove comments from the asm string.  We know that the asmstring only
  // has one line.
  if (!CommentDelimiter.empty() &&
      StringRef(AsmString).find(CommentDelimiter) != StringRef::npos)
    PrintFatalError(TheDef->getLoc(),
                  "asmstring for instruction has comment character in it, "
                  "mark it isCodeGenOnly");

  // Reject matchables with operand modifiers, these aren't something we can
  // handle, the target should be refactored to use operands instead of
  // modifiers.
  //
  // Also, check for instructions which reference the operand multiple times;
  // this implies a constraint we would not honor.
  std::set<std::string> OperandNames;
  for (unsigned i = 0, e = AsmOperands.size(); i != e; ++i) {
    StringRef Tok = AsmOperands[i].Token;
    if (Tok[0] == '$' && Tok.find(':') != StringRef::npos)
      PrintFatalError(TheDef->getLoc(),
                    "matchable with operand modifier '" + Tok.str() +
                    "' not supported by asm matcher.  Mark isCodeGenOnly!");

    // Verify that any operand is only mentioned once.
    // We reject aliases and ignore instructions for now.
    if (Tok[0] == '$' && !OperandNames.insert(Tok).second) {
      if (!Hack)
        PrintFatalError(TheDef->getLoc(),
                      "ERROR: matchable with tied operand '" + Tok.str() +
                      "' can never be matched!");
      // FIXME: Should reject these.  The ARM backend hits this with $lane in a
      // bunch of instructions.  It is unclear what the right answer is.
      DEBUG({
        errs() << "warning: '" << TheDef->getName() << "': "
               << "ignoring instruction with tied operand '"
               << Tok.str() << "'\n";
      });
      return false;
    }
  }

  return true;
}

/// extractSingletonRegisterForAsmOperand - Extract singleton register,
/// if present, from specified token.
void MatchableInfo::
extractSingletonRegisterForAsmOperand(unsigned OperandNo,
                                      const AsmMatcherInfo &Info,
                                      std::string &RegisterPrefix) {
  StringRef Tok = AsmOperands[OperandNo].Token;
  if (RegisterPrefix.empty()) {
    std::string LoweredTok = Tok.lower();
    if (const CodeGenRegister *Reg = Info.Target.getRegisterByName(LoweredTok))
      AsmOperands[OperandNo].SingletonReg = Reg->TheDef;
    return;
  }

  if (!Tok.startswith(RegisterPrefix))
    return;

  StringRef RegName = Tok.substr(RegisterPrefix.size());
  if (const CodeGenRegister *Reg = Info.Target.getRegisterByName(RegName))
    AsmOperands[OperandNo].SingletonReg = Reg->TheDef;

  // If there is no register prefix (i.e. "%" in "%eax"), then this may
  // be some random non-register token, just ignore it.
  return;
}

static std::string getEnumNameForToken(StringRef Str) {
  std::string Res;

  for (StringRef::iterator it = Str.begin(), ie = Str.end(); it != ie; ++it) {
    switch (*it) {
    case '*': Res += "_STAR_"; break;
    case '%': Res += "_PCT_"; break;
    case ':': Res += "_COLON_"; break;
    case '!': Res += "_EXCLAIM_"; break;
    case '.': Res += "_DOT_"; break;
    default:
      if (isalnum(*it))
        Res += *it;
      else
        Res += "_" + utostr((unsigned) *it) + "_";
    }
  }

  return Res;
}

ClassInfo *AsmMatcherInfo::getTokenClass(StringRef Token) {
  ClassInfo *&Entry = TokenClasses[Token];

  if (!Entry) {
    Entry = new ClassInfo();
    Entry->Kind = ClassInfo::Token;
    Entry->ClassName = "Token";
    Entry->Name = "MCK_" + getEnumNameForToken(Token);
    Entry->ValueName = Token;
    Entry->PredicateMethod = "<invalid>";
    Entry->RenderMethod = "<invalid>";
    Entry->ParserMethod = "";
    Entry->DiagnosticType = "";
    Classes.push_back(Entry);
  }

  return Entry;
}

ClassInfo *
AsmMatcherInfo::getOperandClass(const CGIOperandList::OperandInfo &OI,
                                int SubOpIdx) {
  Record *Rec = OI.Rec;
  if (SubOpIdx != -1)
    Rec = cast<DefInit>(OI.MIOperandInfo->getArg(SubOpIdx))->getDef();
  return getOperandClass(Rec, SubOpIdx);
}

ClassInfo *
AsmMatcherInfo::getOperandClass(Record *Rec, int SubOpIdx) {
  if (Rec->isSubClassOf("RegisterOperand")) {
    // RegisterOperand may have an associated ParserMatchClass. If it does,
    // use it, else just fall back to the underlying register class.
    const RecordVal *R = Rec->getValue("ParserMatchClass");
    if (R == 0 || R->getValue() == 0)
      PrintFatalError("Record `" + Rec->getName() +
        "' does not have a ParserMatchClass!\n");

    if (DefInit *DI= dyn_cast<DefInit>(R->getValue())) {
      Record *MatchClass = DI->getDef();
      if (ClassInfo *CI = AsmOperandClasses[MatchClass])
        return CI;
    }

    // No custom match class. Just use the register class.
    Record *ClassRec = Rec->getValueAsDef("RegClass");
    if (!ClassRec)
      PrintFatalError(Rec->getLoc(), "RegisterOperand `" + Rec->getName() +
                    "' has no associated register class!\n");
    if (ClassInfo *CI = RegisterClassClasses[ClassRec])
      return CI;
    PrintFatalError(Rec->getLoc(), "register class has no class info!");
  }


  if (Rec->isSubClassOf("RegisterClass")) {
    if (ClassInfo *CI = RegisterClassClasses[Rec])
      return CI;
    PrintFatalError(Rec->getLoc(), "register class has no class info!");
  }

  if (!Rec->isSubClassOf("Operand"))
    PrintFatalError(Rec->getLoc(), "Operand `" + Rec->getName() +
                  "' does not derive from class Operand!\n");
  Record *MatchClass = Rec->getValueAsDef("ParserMatchClass");
  if (ClassInfo *CI = AsmOperandClasses[MatchClass])
    return CI;

  PrintFatalError(Rec->getLoc(), "operand has no match class!");
}

void AsmMatcherInfo::
buildRegisterClasses(SmallPtrSet<Record*, 16> &SingletonRegisters) {
  const std::vector<CodeGenRegister*> &Registers =
    Target.getRegBank().getRegisters();
  ArrayRef<CodeGenRegisterClass*> RegClassList =
    Target.getRegBank().getRegClasses();

  // The register sets used for matching.
  std::set< std::set<Record*> > RegisterSets;

  // Gather the defined sets.
  for (ArrayRef<CodeGenRegisterClass*>::const_iterator it =
       RegClassList.begin(), ie = RegClassList.end(); it != ie; ++it)
    RegisterSets.insert(std::set<Record*>(
        (*it)->getOrder().begin(), (*it)->getOrder().end()));

  // Add any required singleton sets.
  for (SmallPtrSet<Record*, 16>::iterator it = SingletonRegisters.begin(),
       ie = SingletonRegisters.end(); it != ie; ++it) {
    Record *Rec = *it;
    RegisterSets.insert(std::set<Record*>(&Rec, &Rec + 1));
  }

  // Introduce derived sets where necessary (when a register does not determine
  // a unique register set class), and build the mapping of registers to the set
  // they should classify to.
  std::map<Record*, std::set<Record*> > RegisterMap;
  for (std::vector<CodeGenRegister*>::const_iterator it = Registers.begin(),
         ie = Registers.end(); it != ie; ++it) {
    const CodeGenRegister &CGR = **it;
    // Compute the intersection of all sets containing this register.
    std::set<Record*> ContainingSet;

    for (std::set< std::set<Record*> >::iterator it = RegisterSets.begin(),
           ie = RegisterSets.end(); it != ie; ++it) {
      if (!it->count(CGR.TheDef))
        continue;

      if (ContainingSet.empty()) {
        ContainingSet = *it;
        continue;
      }

      std::set<Record*> Tmp;
      std::swap(Tmp, ContainingSet);
      std::insert_iterator< std::set<Record*> > II(ContainingSet,
                                                   ContainingSet.begin());
      std::set_intersection(Tmp.begin(), Tmp.end(), it->begin(), it->end(), II);
    }

    if (!ContainingSet.empty()) {
      RegisterSets.insert(ContainingSet);
      RegisterMap.insert(std::make_pair(CGR.TheDef, ContainingSet));
    }
  }

  // Construct the register classes.
  std::map<std::set<Record*>, ClassInfo*> RegisterSetClasses;
  unsigned Index = 0;
  for (std::set< std::set<Record*> >::iterator it = RegisterSets.begin(),
         ie = RegisterSets.end(); it != ie; ++it, ++Index) {
    ClassInfo *CI = new ClassInfo();
    CI->Kind = ClassInfo::RegisterClass0 + Index;
    CI->ClassName = "Reg" + utostr(Index);
    CI->Name = "MCK_Reg" + utostr(Index);
    CI->ValueName = "";
    CI->PredicateMethod = ""; // unused
    CI->RenderMethod = "addRegOperands";
    CI->Registers = *it;
    // FIXME: diagnostic type.
    CI->DiagnosticType = "";
    Classes.push_back(CI);
    RegisterSetClasses.insert(std::make_pair(*it, CI));
  }

  // Find the superclasses; we could compute only the subgroup lattice edges,
  // but there isn't really a point.
  for (std::set< std::set<Record*> >::iterator it = RegisterSets.begin(),
         ie = RegisterSets.end(); it != ie; ++it) {
    ClassInfo *CI = RegisterSetClasses[*it];
    for (std::set< std::set<Record*> >::iterator it2 = RegisterSets.begin(),
           ie2 = RegisterSets.end(); it2 != ie2; ++it2)
      if (*it != *it2 &&
          std::includes(it2->begin(), it2->end(), it->begin(), it->end()))
        CI->SuperClasses.push_back(RegisterSetClasses[*it2]);
  }

  // Name the register classes which correspond to a user defined RegisterClass.
  for (ArrayRef<CodeGenRegisterClass*>::const_iterator
       it = RegClassList.begin(), ie = RegClassList.end(); it != ie; ++it) {
    const CodeGenRegisterClass &RC = **it;
    // Def will be NULL for non-user defined register classes.
    Record *Def = RC.getDef();
    if (!Def)
      continue;
    ClassInfo *CI = RegisterSetClasses[std::set<Record*>(RC.getOrder().begin(),
                                                         RC.getOrder().end())];
    if (CI->ValueName.empty()) {
      CI->ClassName = RC.getName();
      CI->Name = "MCK_" + RC.getName();
      CI->ValueName = RC.getName();
    } else
      CI->ValueName = CI->ValueName + "," + RC.getName();

    RegisterClassClasses.insert(std::make_pair(Def, CI));
  }

  // Populate the map for individual registers.
  for (std::map<Record*, std::set<Record*> >::iterator it = RegisterMap.begin(),
         ie = RegisterMap.end(); it != ie; ++it)
    RegisterClasses[it->first] = RegisterSetClasses[it->second];

  // Name the register classes which correspond to singleton registers.
  for (SmallPtrSet<Record*, 16>::iterator it = SingletonRegisters.begin(),
         ie = SingletonRegisters.end(); it != ie; ++it) {
    Record *Rec = *it;
    ClassInfo *CI = RegisterClasses[Rec];
    assert(CI && "Missing singleton register class info!");

    if (CI->ValueName.empty()) {
      CI->ClassName = Rec->getName();
      CI->Name = "MCK_" + Rec->getName();
      CI->ValueName = Rec->getName();
    } else
      CI->ValueName = CI->ValueName + "," + Rec->getName();
  }
}

void AsmMatcherInfo::buildOperandClasses() {
  std::vector<Record*> AsmOperands =
    Records.getAllDerivedDefinitions("AsmOperandClass");

  // Pre-populate AsmOperandClasses map.
  for (std::vector<Record*>::iterator it = AsmOperands.begin(),
         ie = AsmOperands.end(); it != ie; ++it)
    AsmOperandClasses[*it] = new ClassInfo();

  unsigned Index = 0;
  for (std::vector<Record*>::iterator it = AsmOperands.begin(),
         ie = AsmOperands.end(); it != ie; ++it, ++Index) {
    ClassInfo *CI = AsmOperandClasses[*it];
    CI->Kind = ClassInfo::UserClass0 + Index;

    ListInit *Supers = (*it)->getValueAsListInit("SuperClasses");
    for (unsigned i = 0, e = Supers->getSize(); i != e; ++i) {
      DefInit *DI = dyn_cast<DefInit>(Supers->getElement(i));
      if (!DI) {
        PrintError((*it)->getLoc(), "Invalid super class reference!");
        continue;
      }

      ClassInfo *SC = AsmOperandClasses[DI->getDef()];
      if (!SC)
        PrintError((*it)->getLoc(), "Invalid super class reference!");
      else
        CI->SuperClasses.push_back(SC);
    }
    CI->ClassName = (*it)->getValueAsString("Name");
    CI->Name = "MCK_" + CI->ClassName;
    CI->ValueName = (*it)->getName();

    // Get or construct the predicate method name.
    Init *PMName = (*it)->getValueInit("PredicateMethod");
    if (StringInit *SI = dyn_cast<StringInit>(PMName)) {
      CI->PredicateMethod = SI->getValue();
    } else {
      assert(isa<UnsetInit>(PMName) && "Unexpected PredicateMethod field!");
      CI->PredicateMethod = "is" + CI->ClassName;
    }

    // Get or construct the render method name.
    Init *RMName = (*it)->getValueInit("RenderMethod");
    if (StringInit *SI = dyn_cast<StringInit>(RMName)) {
      CI->RenderMethod = SI->getValue();
    } else {
      assert(isa<UnsetInit>(RMName) && "Unexpected RenderMethod field!");
      CI->RenderMethod = "add" + CI->ClassName + "Operands";
    }

    // Get the parse method name or leave it as empty.
    Init *PRMName = (*it)->getValueInit("ParserMethod");
    if (StringInit *SI = dyn_cast<StringInit>(PRMName))
      CI->ParserMethod = SI->getValue();

    // Get the diagnostic type or leave it as empty.
    // Get the parse method name or leave it as empty.
    Init *DiagnosticType = (*it)->getValueInit("DiagnosticType");
    if (StringInit *SI = dyn_cast<StringInit>(DiagnosticType))
      CI->DiagnosticType = SI->getValue();

    AsmOperandClasses[*it] = CI;
    Classes.push_back(CI);
  }
}

AsmMatcherInfo::AsmMatcherInfo(Record *asmParser,
                               CodeGenTarget &target,
                               RecordKeeper &records)
  : Records(records), AsmParser(asmParser), Target(target) {
}

/// buildOperandMatchInfo - Build the necessary information to handle user
/// defined operand parsing methods.
void AsmMatcherInfo::buildOperandMatchInfo() {

  /// Map containing a mask with all operands indices that can be found for
  /// that class inside a instruction.
  typedef std::map<ClassInfo*, unsigned, LessClassInfoPtr> OpClassMaskTy;
  OpClassMaskTy OpClassMask;

  for (std::vector<MatchableInfo*>::const_iterator it =
       Matchables.begin(), ie = Matchables.end();
       it != ie; ++it) {
    MatchableInfo &II = **it;
    OpClassMask.clear();

    // Keep track of all operands of this instructions which belong to the
    // same class.
    for (unsigned i = 0, e = II.AsmOperands.size(); i != e; ++i) {
      MatchableInfo::AsmOperand &Op = II.AsmOperands[i];
      if (Op.Class->ParserMethod.empty())
        continue;
      unsigned &OperandMask = OpClassMask[Op.Class];
      OperandMask |= (1 << i);
    }

    // Generate operand match info for each mnemonic/operand class pair.
    for (OpClassMaskTy::iterator iit = OpClassMask.begin(),
         iie = OpClassMask.end(); iit != iie; ++iit) {
      unsigned OpMask = iit->second;
      ClassInfo *CI = iit->first;
      OperandMatchInfo.push_back(OperandMatchEntry::create(&II, CI, OpMask));
    }
  }
}

void AsmMatcherInfo::buildInfo() {
  // Build information about all of the AssemblerPredicates.
  std::vector<Record*> AllPredicates =
    Records.getAllDerivedDefinitions("Predicate");
  for (unsigned i = 0, e = AllPredicates.size(); i != e; ++i) {
    Record *Pred = AllPredicates[i];
    // Ignore predicates that are not intended for the assembler.
    if (!Pred->getValueAsBit("AssemblerMatcherPredicate"))
      continue;

    if (Pred->getName().empty())
      PrintFatalError(Pred->getLoc(), "Predicate has no name!");

    unsigned FeatureNo = SubtargetFeatures.size();
    SubtargetFeatures[Pred] = new SubtargetFeatureInfo(Pred, FeatureNo);
    assert(FeatureNo < 32 && "Too many subtarget features!");
  }

  // Parse the instructions; we need to do this first so that we can gather the
  // singleton register classes.
  SmallPtrSet<Record*, 16> SingletonRegisters;
  unsigned VariantCount = Target.getAsmParserVariantCount();
  for (unsigned VC = 0; VC != VariantCount; ++VC) {
    Record *AsmVariant = Target.getAsmParserVariant(VC);
    std::string CommentDelimiter =
      AsmVariant->getValueAsString("CommentDelimiter");
    std::string RegisterPrefix = AsmVariant->getValueAsString("RegisterPrefix");
    int AsmVariantNo = AsmVariant->getValueAsInt("Variant");

    for (CodeGenTarget::inst_iterator I = Target.inst_begin(),
           E = Target.inst_end(); I != E; ++I) {
      const CodeGenInstruction &CGI = **I;

      // If the tblgen -match-prefix option is specified (for tblgen hackers),
      // filter the set of instructions we consider.
      if (!StringRef(CGI.TheDef->getName()).startswith(MatchPrefix))
        continue;

      // Ignore "codegen only" instructions.
      if (CGI.TheDef->getValueAsBit("isCodeGenOnly"))
        continue;

      // Validate the operand list to ensure we can handle this instruction.
      for (unsigned i = 0, e = CGI.Operands.size(); i != e; ++i) {
        const CGIOperandList::OperandInfo &OI = CGI.Operands[i];

        // Validate tied operands.
        if (OI.getTiedRegister() != -1) {
          // If we have a tied operand that consists of multiple MCOperands,
          // reject it.  We reject aliases and ignore instructions for now.
          if (OI.MINumOperands != 1) {
            // FIXME: Should reject these.  The ARM backend hits this with $lane
            // in a bunch of instructions. The right answer is unclear.
            DEBUG({
                errs() << "warning: '" << CGI.TheDef->getName() << "': "
                     << "ignoring instruction with multi-operand tied operand '"
                     << OI.Name << "'\n";
              });
            continue;
          }
        }
      }

      OwningPtr<MatchableInfo> II(new MatchableInfo(CGI));

      II->initialize(*this, SingletonRegisters, AsmVariantNo, RegisterPrefix);

      // Ignore instructions which shouldn't be matched and diagnose invalid
      // instruction definitions with an error.
      if (!II->validate(CommentDelimiter, true))
        continue;

      // Ignore "Int_*" and "*_Int" instructions, which are internal aliases.
      //
      // FIXME: This is a total hack.
      if (StringRef(II->TheDef->getName()).startswith("Int_") ||
          StringRef(II->TheDef->getName()).endswith("_Int"))
        continue;

      Matchables.push_back(II.take());
    }

    // Parse all of the InstAlias definitions and stick them in the list of
    // matchables.
    std::vector<Record*> AllInstAliases =
      Records.getAllDerivedDefinitions("InstAlias");
    for (unsigned i = 0, e = AllInstAliases.size(); i != e; ++i) {
      CodeGenInstAlias *Alias = new CodeGenInstAlias(AllInstAliases[i], Target);

      // If the tblgen -match-prefix option is specified (for tblgen hackers),
      // filter the set of instruction aliases we consider, based on the target
      // instruction.
      if (!StringRef(Alias->ResultInst->TheDef->getName())
            .startswith( MatchPrefix))
        continue;

      OwningPtr<MatchableInfo> II(new MatchableInfo(Alias));

      II->initialize(*this, SingletonRegisters, AsmVariantNo, RegisterPrefix);

      // Validate the alias definitions.
      II->validate(CommentDelimiter, false);

      Matchables.push_back(II.take());
    }
  }

  // Build info for the register classes.
  buildRegisterClasses(SingletonRegisters);

  // Build info for the user defined assembly operand classes.
  buildOperandClasses();

  // Build the information about matchables, now that we have fully formed
  // classes.
  std::vector<MatchableInfo*> NewMatchables;
  for (std::vector<MatchableInfo*>::iterator it = Matchables.begin(),
         ie = Matchables.end(); it != ie; ++it) {
    MatchableInfo *II = *it;

    // Parse the tokens after the mnemonic.
    // Note: buildInstructionOperandReference may insert new AsmOperands, so
    // don't precompute the loop bound.
    for (unsigned i = 0; i != II->AsmOperands.size(); ++i) {
      MatchableInfo::AsmOperand &Op = II->AsmOperands[i];
      StringRef Token = Op.Token;

      // Check for singleton registers.
      if (Record *RegRecord = II->AsmOperands[i].SingletonReg) {
        Op.Class = RegisterClasses[RegRecord];
        assert(Op.Class && Op.Class->Registers.size() == 1 &&
               "Unexpected class for singleton register");
        continue;
      }

      // Check for simple tokens.
      if (Token[0] != '$') {
        Op.Class = getTokenClass(Token);
        continue;
      }

      if (Token.size() > 1 && isdigit(Token[1])) {
        Op.Class = getTokenClass(Token);
        continue;
      }

      // Otherwise this is an operand reference.
      StringRef OperandName;
      if (Token[1] == '{')
        OperandName = Token.substr(2, Token.size() - 3);
      else
        OperandName = Token.substr(1);

      if (II->DefRec.is<const CodeGenInstruction*>())
        buildInstructionOperandReference(II, OperandName, i);
      else
        buildAliasOperandReference(II, OperandName, Op);
    }

    if (II->DefRec.is<const CodeGenInstruction*>()) {
      II->buildInstructionResultOperands();
      // If the instruction has a two-operand alias, build up the
      // matchable here. We'll add them in bulk at the end to avoid
      // confusing this loop.
      std::string Constraint =
        II->TheDef->getValueAsString("TwoOperandAliasConstraint");
      if (Constraint != "") {
        // Start by making a copy of the original matchable.
        OwningPtr<MatchableInfo> AliasII(new MatchableInfo(*II));

        // Adjust it to be a two-operand alias.
        AliasII->formTwoOperandAlias(Constraint);

        // Add the alias to the matchables list.
        NewMatchables.push_back(AliasII.take());
      }
    } else
      II->buildAliasResultOperands();
  }
  if (!NewMatchables.empty())
    Matchables.insert(Matchables.end(), NewMatchables.begin(),
                      NewMatchables.end());

  // Process token alias definitions and set up the associated superclass
  // information.
  std::vector<Record*> AllTokenAliases =
    Records.getAllDerivedDefinitions("TokenAlias");
  for (unsigned i = 0, e = AllTokenAliases.size(); i != e; ++i) {
    Record *Rec = AllTokenAliases[i];
    ClassInfo *FromClass = getTokenClass(Rec->getValueAsString("FromToken"));
    ClassInfo *ToClass = getTokenClass(Rec->getValueAsString("ToToken"));
    if (FromClass == ToClass)
      PrintFatalError(Rec->getLoc(),
                    "error: Destination value identical to source value.");
    FromClass->SuperClasses.push_back(ToClass);
  }

  // Reorder classes so that classes precede super classes.
  std::sort(Classes.begin(), Classes.end(), less_ptr<ClassInfo>());
}

/// buildInstructionOperandReference - The specified operand is a reference to a
/// named operand such as $src.  Resolve the Class and OperandInfo pointers.
void AsmMatcherInfo::
buildInstructionOperandReference(MatchableInfo *II,
                                 StringRef OperandName,
                                 unsigned AsmOpIdx) {
  const CodeGenInstruction &CGI = *II->DefRec.get<const CodeGenInstruction*>();
  const CGIOperandList &Operands = CGI.Operands;
  MatchableInfo::AsmOperand *Op = &II->AsmOperands[AsmOpIdx];

  // Map this token to an operand.
  unsigned Idx;
  if (!Operands.hasOperandNamed(OperandName, Idx))
    PrintFatalError(II->TheDef->getLoc(), "error: unable to find operand: '" +
                  OperandName.str() + "'");

  // If the instruction operand has multiple suboperands, but the parser
  // match class for the asm operand is still the default "ImmAsmOperand",
  // then handle each suboperand separately.
  if (Op->SubOpIdx == -1 && Operands[Idx].MINumOperands > 1) {
    Record *Rec = Operands[Idx].Rec;
    assert(Rec->isSubClassOf("Operand") && "Unexpected operand!");
    Record *MatchClass = Rec->getValueAsDef("ParserMatchClass");
    if (MatchClass && MatchClass->getValueAsString("Name") == "Imm") {
      // Insert remaining suboperands after AsmOpIdx in II->AsmOperands.
      StringRef Token = Op->Token; // save this in case Op gets moved
      for (unsigned SI = 1, SE = Operands[Idx].MINumOperands; SI != SE; ++SI) {
        MatchableInfo::AsmOperand NewAsmOp(Token);
        NewAsmOp.SubOpIdx = SI;
        II->AsmOperands.insert(II->AsmOperands.begin()+AsmOpIdx+SI, NewAsmOp);
      }
      // Replace Op with first suboperand.
      Op = &II->AsmOperands[AsmOpIdx]; // update the pointer in case it moved
      Op->SubOpIdx = 0;
    }
  }

  // Set up the operand class.
  Op->Class = getOperandClass(Operands[Idx], Op->SubOpIdx);

  // If the named operand is tied, canonicalize it to the untied operand.
  // For example, something like:
  //   (outs GPR:$dst), (ins GPR:$src)
  // with an asmstring of
  //   "inc $src"
  // we want to canonicalize to:
  //   "inc $dst"
  // so that we know how to provide the $dst operand when filling in the result.
  int OITied = Operands[Idx].getTiedRegister();
  if (OITied != -1) {
    // The tied operand index is an MIOperand index, find the operand that
    // contains it.
    std::pair<unsigned, unsigned> Idx = Operands.getSubOperandNumber(OITied);
    OperandName = Operands[Idx.first].Name;
    Op->SubOpIdx = Idx.second;
  }

  Op->SrcOpName = OperandName;
}

/// buildAliasOperandReference - When parsing an operand reference out of the
/// matching string (e.g. "movsx $src, $dst"), determine what the class of the
/// operand reference is by looking it up in the result pattern definition.
void AsmMatcherInfo::buildAliasOperandReference(MatchableInfo *II,
                                                StringRef OperandName,
                                                MatchableInfo::AsmOperand &Op) {
  const CodeGenInstAlias &CGA = *II->DefRec.get<const CodeGenInstAlias*>();

  // Set up the operand class.
  for (unsigned i = 0, e = CGA.ResultOperands.size(); i != e; ++i)
    if (CGA.ResultOperands[i].isRecord() &&
        CGA.ResultOperands[i].getName() == OperandName) {
      // It's safe to go with the first one we find, because CodeGenInstAlias
      // validates that all operands with the same name have the same record.
      Op.SubOpIdx = CGA.ResultInstOperandIndex[i].second;
      // Use the match class from the Alias definition, not the
      // destination instruction, as we may have an immediate that's
      // being munged by the match class.
      Op.Class = getOperandClass(CGA.ResultOperands[i].getRecord(),
                                 Op.SubOpIdx);
      Op.SrcOpName = OperandName;
      return;
    }

  PrintFatalError(II->TheDef->getLoc(), "error: unable to find operand: '" +
                OperandName.str() + "'");
}

void MatchableInfo::buildInstructionResultOperands() {
  const CodeGenInstruction *ResultInst = getResultInst();

  // Loop over all operands of the result instruction, determining how to
  // populate them.
  for (unsigned i = 0, e = ResultInst->Operands.size(); i != e; ++i) {
    const CGIOperandList::OperandInfo &OpInfo = ResultInst->Operands[i];

    // If this is a tied operand, just copy from the previously handled operand.
    int TiedOp = OpInfo.getTiedRegister();
    if (TiedOp != -1) {
      ResOperands.push_back(ResOperand::getTiedOp(TiedOp));
      continue;
    }

    // Find out what operand from the asmparser this MCInst operand comes from.
    int SrcOperand = findAsmOperandNamed(OpInfo.Name);
    if (OpInfo.Name.empty() || SrcOperand == -1)
      PrintFatalError(TheDef->getLoc(), "Instruction '" +
                    TheDef->getName() + "' has operand '" + OpInfo.Name +
                    "' that doesn't appear in asm string!");

    // Check if the one AsmOperand populates the entire operand.
    unsigned NumOperands = OpInfo.MINumOperands;
    if (AsmOperands[SrcOperand].SubOpIdx == -1) {
      ResOperands.push_back(ResOperand::getRenderedOp(SrcOperand, NumOperands));
      continue;
    }

    // Add a separate ResOperand for each suboperand.
    for (unsigned AI = 0; AI < NumOperands; ++AI) {
      assert(AsmOperands[SrcOperand+AI].SubOpIdx == (int)AI &&
             AsmOperands[SrcOperand+AI].SrcOpName == OpInfo.Name &&
             "unexpected AsmOperands for suboperands");
      ResOperands.push_back(ResOperand::getRenderedOp(SrcOperand + AI, 1));
    }
  }
}

void MatchableInfo::buildAliasResultOperands() {
  const CodeGenInstAlias &CGA = *DefRec.get<const CodeGenInstAlias*>();
  const CodeGenInstruction *ResultInst = getResultInst();

  // Loop over all operands of the result instruction, determining how to
  // populate them.
  unsigned AliasOpNo = 0;
  unsigned LastOpNo = CGA.ResultInstOperandIndex.size();
  for (unsigned i = 0, e = ResultInst->Operands.size(); i != e; ++i) {
    const CGIOperandList::OperandInfo *OpInfo = &ResultInst->Operands[i];

    // If this is a tied operand, just copy from the previously handled operand.
    int TiedOp = OpInfo->getTiedRegister();
    if (TiedOp != -1) {
      ResOperands.push_back(ResOperand::getTiedOp(TiedOp));
      continue;
    }

    // Handle all the suboperands for this operand.
    const std::string &OpName = OpInfo->Name;
    for ( ; AliasOpNo <  LastOpNo &&
            CGA.ResultInstOperandIndex[AliasOpNo].first == i; ++AliasOpNo) {
      int SubIdx = CGA.ResultInstOperandIndex[AliasOpNo].second;

      // Find out what operand from the asmparser that this MCInst operand
      // comes from.
      switch (CGA.ResultOperands[AliasOpNo].Kind) {
      case CodeGenInstAlias::ResultOperand::K_Record: {
        StringRef Name = CGA.ResultOperands[AliasOpNo].getName();
        int SrcOperand = findAsmOperand(Name, SubIdx);
        if (SrcOperand == -1)
          PrintFatalError(TheDef->getLoc(), "Instruction '" +
                        TheDef->getName() + "' has operand '" + OpName +
                        "' that doesn't appear in asm string!");
        unsigned NumOperands = (SubIdx == -1 ? OpInfo->MINumOperands : 1);
        ResOperands.push_back(ResOperand::getRenderedOp(SrcOperand,
                                                        NumOperands));
        break;
      }
      case CodeGenInstAlias::ResultOperand::K_Imm: {
        int64_t ImmVal = CGA.ResultOperands[AliasOpNo].getImm();
        ResOperands.push_back(ResOperand::getImmOp(ImmVal));
        break;
      }
      case CodeGenInstAlias::ResultOperand::K_Reg: {
        Record *Reg = CGA.ResultOperands[AliasOpNo].getRegister();
        ResOperands.push_back(ResOperand::getRegOp(Reg));
        break;
      }
      }
    }
  }
}

static unsigned getConverterOperandID(const std::string &Name,
                                      SetVector<std::string> &Table,
                                      bool &IsNew) {
  IsNew = Table.insert(Name);

  unsigned ID = IsNew ? Table.size() - 1 :
    std::find(Table.begin(), Table.end(), Name) - Table.begin();

  assert(ID < Table.size());

  return ID;
}


static void emitConvertFuncs(CodeGenTarget &Target, StringRef ClassName,
                             std::vector<MatchableInfo*> &Infos,
                             raw_ostream &OS) {
  SetVector<std::string> OperandConversionKinds;
  SetVector<std::string> InstructionConversionKinds;
  std::vector<std::vector<uint8_t> > ConversionTable;
  size_t MaxRowLength = 2; // minimum is custom converter plus terminator.

  // TargetOperandClass - This is the target's operand class, like X86Operand.
  std::string TargetOperandClass = Target.getName() + "Operand";

  // Write the convert function to a separate stream, so we can drop it after
  // the enum. We'll build up the conversion handlers for the individual
  // operand types opportunistically as we encounter them.
  std::string ConvertFnBody;
  raw_string_ostream CvtOS(ConvertFnBody);
  // Start the unified conversion function.
  CvtOS << "void " << Target.getName() << ClassName << "::\n"
        << "convertToMCInst(unsigned Kind, MCInst &Inst, "
        << "unsigned Opcode,\n"
        << "                const SmallVectorImpl<MCParsedAsmOperand*"
        << "> &Operands) {\n"
        << "  assert(Kind < CVT_NUM_SIGNATURES && \"Invalid signature!\");\n"
        << "  const uint8_t *Converter = ConversionTable[Kind];\n"
        << "  Inst.setOpcode(Opcode);\n"
        << "  for (const uint8_t *p = Converter; *p; p+= 2) {\n"
        << "    switch (*p) {\n"
        << "    default: llvm_unreachable(\"invalid conversion entry!\");\n"
        << "    case CVT_Reg:\n"
        << "      static_cast<" << TargetOperandClass
        << "*>(Operands[*(p + 1)])->addRegOperands(Inst, 1);\n"
        << "      break;\n"
        << "    case CVT_Tied:\n"
        << "      Inst.addOperand(Inst.getOperand(*(p + 1)));\n"
        << "      break;\n";

  std::string OperandFnBody;
  raw_string_ostream OpOS(OperandFnBody);
  // Start the operand number lookup function.
  OpOS << "void " << Target.getName() << ClassName << "::\n"
       << "convertToMapAndConstraints(unsigned Kind,\n";
  OpOS.indent(27);
  OpOS << "const SmallVectorImpl<MCParsedAsmOperand*> &Operands) {\n"
       << "  assert(Kind < CVT_NUM_SIGNATURES && \"Invalid signature!\");\n"
       << "  unsigned NumMCOperands = 0;\n"
       << "  const uint8_t *Converter = ConversionTable[Kind];\n"
       << "  for (const uint8_t *p = Converter; *p; p+= 2) {\n"
       << "    switch (*p) {\n"
       << "    default: llvm_unreachable(\"invalid conversion entry!\");\n"
       << "    case CVT_Reg:\n"
       << "      Operands[*(p + 1)]->setMCOperandNum(NumMCOperands);\n"
       << "      Operands[*(p + 1)]->setConstraint(\"m\");\n"
       << "      ++NumMCOperands;\n"
       << "      break;\n"
       << "    case CVT_Tied:\n"
       << "      ++NumMCOperands;\n"
       << "      break;\n";

  // Pre-populate the operand conversion kinds with the standard always
  // available entries.
  OperandConversionKinds.insert("CVT_Done");
  OperandConversionKinds.insert("CVT_Reg");
  OperandConversionKinds.insert("CVT_Tied");
  enum { CVT_Done, CVT_Reg, CVT_Tied };

  for (std::vector<MatchableInfo*>::const_iterator it = Infos.begin(),
         ie = Infos.end(); it != ie; ++it) {
    MatchableInfo &II = **it;

    // Check if we have a custom match function.
    std::string AsmMatchConverter =
      II.getResultInst()->TheDef->getValueAsString("AsmMatchConverter");
    if (!AsmMatchConverter.empty()) {
      std::string Signature = "ConvertCustom_" + AsmMatchConverter;
      II.ConversionFnKind = Signature;

      // Check if we have already generated this signature.
      if (!InstructionConversionKinds.insert(Signature))
        continue;

      // Remember this converter for the kind enum.
      unsigned KindID = OperandConversionKinds.size();
      OperandConversionKinds.insert("CVT_" + AsmMatchConverter);

      // Add the converter row for this instruction.
      ConversionTable.push_back(std::vector<uint8_t>());
      ConversionTable.back().push_back(KindID);
      ConversionTable.back().push_back(CVT_Done);

      // Add the handler to the conversion driver function.
      CvtOS << "    case CVT_" << AsmMatchConverter << ":\n"
            << "      " << AsmMatchConverter << "(Inst, Operands);\n"
            << "      break;\n";

      // FIXME: Handle the operand number lookup for custom match functions.
      continue;
    }

    // Build the conversion function signature.
    std::string Signature = "Convert";

    std::vector<uint8_t> ConversionRow;

    // Compute the convert enum and the case body.
    MaxRowLength = std::max(MaxRowLength, II.ResOperands.size()*2 + 1 );

    for (unsigned i = 0, e = II.ResOperands.size(); i != e; ++i) {
      const MatchableInfo::ResOperand &OpInfo = II.ResOperands[i];

      // Generate code to populate each result operand.
      switch (OpInfo.Kind) {
      case MatchableInfo::ResOperand::RenderAsmOperand: {
        // This comes from something we parsed.
        MatchableInfo::AsmOperand &Op = II.AsmOperands[OpInfo.AsmOperandNum];

        // Registers are always converted the same, don't duplicate the
        // conversion function based on them.
        Signature += "__";
        std::string Class;
        Class = Op.Class->isRegisterClass() ? "Reg" : Op.Class->ClassName;
        Signature += Class;
        Signature += utostr(OpInfo.MINumOperands);
        Signature += "_" + itostr(OpInfo.AsmOperandNum);

        // Add the conversion kind, if necessary, and get the associated ID
        // the index of its entry in the vector).
        std::string Name = "CVT_" + (Op.Class->isRegisterClass() ? "Reg" :
                                     Op.Class->RenderMethod);

        bool IsNewConverter = false;
        unsigned ID = getConverterOperandID(Name, OperandConversionKinds,
                                            IsNewConverter);

        // Add the operand entry to the instruction kind conversion row.
        ConversionRow.push_back(ID);
        ConversionRow.push_back(OpInfo.AsmOperandNum + 1);

        if (!IsNewConverter)
          break;

        // This is a new operand kind. Add a handler for it to the
        // converter driver.
        CvtOS << "    case " << Name << ":\n"
              << "      static_cast<" << TargetOperandClass
              << "*>(Operands[*(p + 1)])->"
              << Op.Class->RenderMethod << "(Inst, " << OpInfo.MINumOperands
              << ");\n"
              << "      break;\n";

        // Add a handler for the operand number lookup.
        OpOS << "    case " << Name << ":\n"
             << "      Operands[*(p + 1)]->setMCOperandNum(NumMCOperands);\n"
             << "      Operands[*(p + 1)]->setConstraint(\"m\");\n"
             << "      NumMCOperands += " << OpInfo.MINumOperands << ";\n"
             << "      break;\n";
        break;
      }
      case MatchableInfo::ResOperand::TiedOperand: {
        // If this operand is tied to a previous one, just copy the MCInst
        // operand from the earlier one.We can only tie single MCOperand values.
        //assert(OpInfo.MINumOperands == 1 && "Not a singular MCOperand");
        unsigned TiedOp = OpInfo.TiedOperandNum;
        assert(i > TiedOp && "Tied operand precedes its target!");
        Signature += "__Tie" + utostr(TiedOp);
        ConversionRow.push_back(CVT_Tied);
        ConversionRow.push_back(TiedOp);
        // FIXME: Handle the operand number lookup for tied operands.
        break;
      }
      case MatchableInfo::ResOperand::ImmOperand: {
        int64_t Val = OpInfo.ImmVal;
        std::string Ty = "imm_" + itostr(Val);
        Signature += "__" + Ty;

        std::string Name = "CVT_" + Ty;
        bool IsNewConverter = false;
        unsigned ID = getConverterOperandID(Name, OperandConversionKinds,
                                            IsNewConverter);
        // Add the operand entry to the instruction kind conversion row.
        ConversionRow.push_back(ID);
        ConversionRow.push_back(0);

        if (!IsNewConverter)
          break;

        CvtOS << "    case " << Name << ":\n"
              << "      Inst.addOperand(MCOperand::CreateImm(" << Val << "));\n"
              << "      break;\n";

        OpOS << "    case " << Name << ":\n"
             << "      Operands[*(p + 1)]->setMCOperandNum(NumMCOperands);\n"
             << "      Operands[*(p + 1)]->setConstraint(\"\");\n"
             << "      ++NumMCOperands;\n"
             << "      break;\n";
        break;
      }
      case MatchableInfo::ResOperand::RegOperand: {
        std::string Reg, Name;
        if (OpInfo.Register == 0) {
          Name = "reg0";
          Reg = "0";
        } else {
          Reg = getQualifiedName(OpInfo.Register);
          Name = "reg" + OpInfo.Register->getName();
        }
        Signature += "__" + Name;
        Name = "CVT_" + Name;
        bool IsNewConverter = false;
        unsigned ID = getConverterOperandID(Name, OperandConversionKinds,
                                            IsNewConverter);
        // Add the operand entry to the instruction kind conversion row.
        ConversionRow.push_back(ID);
        ConversionRow.push_back(0);

        if (!IsNewConverter)
          break;
        CvtOS << "    case " << Name << ":\n"
              << "      Inst.addOperand(MCOperand::CreateReg(" << Reg << "));\n"
              << "      break;\n";

        OpOS << "    case " << Name << ":\n"
             << "      Operands[*(p + 1)]->setMCOperandNum(NumMCOperands);\n"
             << "      Operands[*(p + 1)]->setConstraint(\"m\");\n"
             << "      ++NumMCOperands;\n"
             << "      break;\n";
      }
      }
    }

    // If there were no operands, add to the signature to that effect
    if (Signature == "Convert")
      Signature += "_NoOperands";

    II.ConversionFnKind = Signature;

    // Save the signature. If we already have it, don't add a new row
    // to the table.
    if (!InstructionConversionKinds.insert(Signature))
      continue;

    // Add the row to the table.
    ConversionTable.push_back(ConversionRow);
  }

  // Finish up the converter driver function.
  CvtOS << "    }\n  }\n}\n\n";

  // Finish up the operand number lookup function.
  OpOS << "    }\n  }\n}\n\n";

  OS << "namespace {\n";

  // Output the operand conversion kind enum.
  OS << "enum OperatorConversionKind {\n";
  for (unsigned i = 0, e = OperandConversionKinds.size(); i != e; ++i)
    OS << "  " << OperandConversionKinds[i] << ",\n";
  OS << "  CVT_NUM_CONVERTERS\n";
  OS << "};\n\n";

  // Output the instruction conversion kind enum.
  OS << "enum InstructionConversionKind {\n";
  for (SetVector<std::string>::const_iterator
         i = InstructionConversionKinds.begin(),
         e = InstructionConversionKinds.end(); i != e; ++i)
    OS << "  " << *i << ",\n";
  OS << "  CVT_NUM_SIGNATURES\n";
  OS << "};\n\n";


  OS << "} // end anonymous namespace\n\n";

  // Output the conversion table.
  OS << "static const uint8_t ConversionTable[CVT_NUM_SIGNATURES]["
     << MaxRowLength << "] = {\n";

  for (unsigned Row = 0, ERow = ConversionTable.size(); Row != ERow; ++Row) {
    assert(ConversionTable[Row].size() % 2 == 0 && "bad conversion row!");
    OS << "  // " << InstructionConversionKinds[Row] << "\n";
    OS << "  { ";
    for (unsigned i = 0, e = ConversionTable[Row].size(); i != e; i += 2)
      OS << OperandConversionKinds[ConversionTable[Row][i]] << ", "
         << (unsigned)(ConversionTable[Row][i + 1]) << ", ";
    OS << "CVT_Done },\n";
  }

  OS << "};\n\n";

  // Spit out the conversion driver function.
  OS << CvtOS.str();

  // Spit out the operand number lookup function.
  OS << OpOS.str();
}

/// emitMatchClassEnumeration - Emit the enumeration for match class kinds.
static void emitMatchClassEnumeration(CodeGenTarget &Target,
                                      std::vector<ClassInfo*> &Infos,
                                      raw_ostream &OS) {
  OS << "namespace {\n\n";

  OS << "/// MatchClassKind - The kinds of classes which participate in\n"
     << "/// instruction matching.\n";
  OS << "enum MatchClassKind {\n";
  OS << "  InvalidMatchClass = 0,\n";
  for (std::vector<ClassInfo*>::iterator it = Infos.begin(),
         ie = Infos.end(); it != ie; ++it) {
    ClassInfo &CI = **it;
    OS << "  " << CI.Name << ", // ";
    if (CI.Kind == ClassInfo::Token) {
      OS << "'" << CI.ValueName << "'\n";
    } else if (CI.isRegisterClass()) {
      if (!CI.ValueName.empty())
        OS << "register class '" << CI.ValueName << "'\n";
      else
        OS << "derived register class\n";
    } else {
      OS << "user defined class '" << CI.ValueName << "'\n";
    }
  }
  OS << "  NumMatchClassKinds\n";
  OS << "};\n\n";

  OS << "}\n\n";
}

/// emitValidateOperandClass - Emit the function to validate an operand class.
static void emitValidateOperandClass(AsmMatcherInfo &Info,
                                     raw_ostream &OS) {
  OS << "static unsigned validateOperandClass(MCParsedAsmOperand *GOp, "
     << "MatchClassKind Kind) {\n";
  OS << "  " << Info.Target.getName() << "Operand &Operand = *("
     << Info.Target.getName() << "Operand*)GOp;\n";

  // The InvalidMatchClass is not to match any operand.
  OS << "  if (Kind == InvalidMatchClass)\n";
  OS << "    return MCTargetAsmParser::Match_InvalidOperand;\n\n";

  // Check for Token operands first.
  // FIXME: Use a more specific diagnostic type.
  OS << "  if (Operand.isToken())\n";
  OS << "    return isSubclass(matchTokenString(Operand.getToken()), Kind) ?\n"
     << "             MCTargetAsmParser::Match_Success :\n"
     << "             MCTargetAsmParser::Match_InvalidOperand;\n\n";

  // Check the user classes. We don't care what order since we're only
  // actually matching against one of them.
  for (std::vector<ClassInfo*>::iterator it = Info.Classes.begin(),
         ie = Info.Classes.end(); it != ie; ++it) {
    ClassInfo &CI = **it;

    if (!CI.isUserClass())
      continue;

    OS << "  // '" << CI.ClassName << "' class\n";
    OS << "  if (Kind == " << CI.Name << ") {\n";
    OS << "    if (Operand." << CI.PredicateMethod << "())\n";
    OS << "      return MCTargetAsmParser::Match_Success;\n";
    if (!CI.DiagnosticType.empty())
      OS << "    return " << Info.Target.getName() << "AsmParser::Match_"
         << CI.DiagnosticType << ";\n";
    OS << "  }\n\n";
  }

  // Check for register operands, including sub-classes.
  OS << "  if (Operand.isReg()) {\n";
  OS << "    MatchClassKind OpKind;\n";
  OS << "    switch (Operand.getReg()) {\n";
  OS << "    default: OpKind = InvalidMatchClass; break;\n";
  for (AsmMatcherInfo::RegisterClassesTy::iterator
         it = Info.RegisterClasses.begin(), ie = Info.RegisterClasses.end();
       it != ie; ++it)
    OS << "    case " << Info.Target.getName() << "::"
       << it->first->getName() << ": OpKind = " << it->second->Name
       << "; break;\n";
  OS << "    }\n";
  OS << "    return isSubclass(OpKind, Kind) ? "
     << "MCTargetAsmParser::Match_Success :\n                             "
     << "         MCTargetAsmParser::Match_InvalidOperand;\n  }\n\n";

  // Generic fallthrough match failure case for operands that don't have
  // specialized diagnostic types.
  OS << "  return MCTargetAsmParser::Match_InvalidOperand;\n";
  OS << "}\n\n";
}

/// emitIsSubclass - Emit the subclass predicate function.
static void emitIsSubclass(CodeGenTarget &Target,
                           std::vector<ClassInfo*> &Infos,
                           raw_ostream &OS) {
  OS << "/// isSubclass - Compute whether \\p A is a subclass of \\p B.\n";
  OS << "static bool isSubclass(MatchClassKind A, MatchClassKind B) {\n";
  OS << "  if (A == B)\n";
  OS << "    return true;\n\n";

  OS << "  switch (A) {\n";
  OS << "  default:\n";
  OS << "    return false;\n";
  for (std::vector<ClassInfo*>::iterator it = Infos.begin(),
         ie = Infos.end(); it != ie; ++it) {
    ClassInfo &A = **it;

    std::vector<StringRef> SuperClasses;
    for (std::vector<ClassInfo*>::iterator it = Infos.begin(),
         ie = Infos.end(); it != ie; ++it) {
      ClassInfo &B = **it;

      if (&A != &B && A.isSubsetOf(B))
        SuperClasses.push_back(B.Name);
    }

    if (SuperClasses.empty())
      continue;

    OS << "\n  case " << A.Name << ":\n";

    if (SuperClasses.size() == 1) {
      OS << "    return B == " << SuperClasses.back() << ";\n";
      continue;
    }

    OS << "    switch (B) {\n";
    OS << "    default: return false;\n";
    for (unsigned i = 0, e = SuperClasses.size(); i != e; ++i)
      OS << "    case " << SuperClasses[i] << ": return true;\n";
    OS << "    }\n";
  }
  OS << "  }\n";
  OS << "}\n\n";
}

/// emitMatchTokenString - Emit the function to match a token string to the
/// appropriate match class value.
static void emitMatchTokenString(CodeGenTarget &Target,
                                 std::vector<ClassInfo*> &Infos,
                                 raw_ostream &OS) {
  // Construct the match list.
  std::vector<StringMatcher::StringPair> Matches;
  for (std::vector<ClassInfo*>::iterator it = Infos.begin(),
         ie = Infos.end(); it != ie; ++it) {
    ClassInfo &CI = **it;

    if (CI.Kind == ClassInfo::Token)
      Matches.push_back(StringMatcher::StringPair(CI.ValueName,
                                                  "return " + CI.Name + ";"));
  }

  OS << "static MatchClassKind matchTokenString(StringRef Name) {\n";

  StringMatcher("Name", Matches, OS).Emit();

  OS << "  return InvalidMatchClass;\n";
  OS << "}\n\n";
}

/// emitMatchRegisterName - Emit the function to match a string to the target
/// specific register enum.
static void emitMatchRegisterName(CodeGenTarget &Target, Record *AsmParser,
                                  raw_ostream &OS) {
  // Construct the match list.
  std::vector<StringMatcher::StringPair> Matches;
  const std::vector<CodeGenRegister*> &Regs =
    Target.getRegBank().getRegisters();
  for (unsigned i = 0, e = Regs.size(); i != e; ++i) {
    const CodeGenRegister *Reg = Regs[i];
    if (Reg->TheDef->getValueAsString("AsmName").empty())
      continue;

    Matches.push_back(StringMatcher::StringPair(
                                     Reg->TheDef->getValueAsString("AsmName"),
                                     "return " + utostr(Reg->EnumValue) + ";"));
  }

  OS << "static unsigned MatchRegisterName(StringRef Name) {\n";

  StringMatcher("Name", Matches, OS).Emit();

  OS << "  return 0;\n";
  OS << "}\n\n";
}

/// emitSubtargetFeatureFlagEnumeration - Emit the subtarget feature flag
/// definitions.
static void emitSubtargetFeatureFlagEnumeration(AsmMatcherInfo &Info,
                                                raw_ostream &OS) {
  OS << "// Flags for subtarget features that participate in "
     << "instruction matching.\n";
  OS << "enum SubtargetFeatureFlag {\n";
  for (std::map<Record*, SubtargetFeatureInfo*>::const_iterator
         it = Info.SubtargetFeatures.begin(),
         ie = Info.SubtargetFeatures.end(); it != ie; ++it) {
    SubtargetFeatureInfo &SFI = *it->second;
    OS << "  " << SFI.getEnumName() << " = (1 << " << SFI.Index << "),\n";
  }
  OS << "  Feature_None = 0\n";
  OS << "};\n\n";
}

/// emitOperandDiagnosticTypes - Emit the operand matching diagnostic types.
static void emitOperandDiagnosticTypes(AsmMatcherInfo &Info, raw_ostream &OS) {
  // Get the set of diagnostic types from all of the operand classes.
  std::set<StringRef> Types;
  for (std::map<Record*, ClassInfo*>::const_iterator
       I = Info.AsmOperandClasses.begin(),
       E = Info.AsmOperandClasses.end(); I != E; ++I) {
    if (!I->second->DiagnosticType.empty())
      Types.insert(I->second->DiagnosticType);
  }

  if (Types.empty()) return;

  // Now emit the enum entries.
  for (std::set<StringRef>::const_iterator I = Types.begin(), E = Types.end();
       I != E; ++I)
    OS << "  Match_" << *I << ",\n";
  OS << "  END_OPERAND_DIAGNOSTIC_TYPES\n";
}

/// emitGetSubtargetFeatureName - Emit the helper function to get the
/// user-level name for a subtarget feature.
static void emitGetSubtargetFeatureName(AsmMatcherInfo &Info, raw_ostream &OS) {
  OS << "// User-level names for subtarget features that participate in\n"
     << "// instruction matching.\n"
     << "static const char *getSubtargetFeatureName(unsigned Val) {\n"
     << "  switch(Val) {\n";
  for (std::map<Record*, SubtargetFeatureInfo*>::const_iterator
         it = Info.SubtargetFeatures.begin(),
         ie = Info.SubtargetFeatures.end(); it != ie; ++it) {
    SubtargetFeatureInfo &SFI = *it->second;
    // FIXME: Totally just a placeholder name to get the algorithm working.
    OS << "  case " << SFI.getEnumName() << ": return \""
       << SFI.TheDef->getValueAsString("PredicateName") << "\";\n";
  }
  OS << "  default: return \"(unknown)\";\n";
  OS << "  }\n}\n\n";
}

/// emitComputeAvailableFeatures - Emit the function to compute the list of
/// available features given a subtarget.
static void emitComputeAvailableFeatures(AsmMatcherInfo &Info,
                                         raw_ostream &OS) {
  std::string ClassName =
    Info.AsmParser->getValueAsString("AsmParserClassName");

  OS << "unsigned " << Info.Target.getName() << ClassName << "::\n"
     << "ComputeAvailableFeatures(uint64_t FB) const {\n";
  OS << "  unsigned Features = 0;\n";
  for (std::map<Record*, SubtargetFeatureInfo*>::const_iterator
         it = Info.SubtargetFeatures.begin(),
         ie = Info.SubtargetFeatures.end(); it != ie; ++it) {
    SubtargetFeatureInfo &SFI = *it->second;

    OS << "  if (";
    std::string CondStorage =
      SFI.TheDef->getValueAsString("AssemblerCondString");
    StringRef Conds = CondStorage;
    std::pair<StringRef,StringRef> Comma = Conds.split(',');
    bool First = true;
    do {
      if (!First)
        OS << " && ";

      bool Neg = false;
      StringRef Cond = Comma.first;
      if (Cond[0] == '!') {
        Neg = true;
        Cond = Cond.substr(1);
      }

      OS << "((FB & " << Info.Target.getName() << "::" << Cond << ")";
      if (Neg)
        OS << " == 0";
      else
        OS << " != 0";
      OS << ")";

      if (Comma.second.empty())
        break;

      First = false;
      Comma = Comma.second.split(',');
    } while (true);

    OS << ")\n";
    OS << "    Features |= " << SFI.getEnumName() << ";\n";
  }
  OS << "  return Features;\n";
  OS << "}\n\n";
}

static std::string GetAliasRequiredFeatures(Record *R,
                                            const AsmMatcherInfo &Info) {
  std::vector<Record*> ReqFeatures = R->getValueAsListOfDefs("Predicates");
  std::string Result;
  unsigned NumFeatures = 0;
  for (unsigned i = 0, e = ReqFeatures.size(); i != e; ++i) {
    SubtargetFeatureInfo *F = Info.getSubtargetFeature(ReqFeatures[i]);

    if (F == 0)
      PrintFatalError(R->getLoc(), "Predicate '" + ReqFeatures[i]->getName() +
                    "' is not marked as an AssemblerPredicate!");

    if (NumFeatures)
      Result += '|';

    Result += F->getEnumName();
    ++NumFeatures;
  }

  if (NumFeatures > 1)
    Result = '(' + Result + ')';
  return Result;
}

/// emitMnemonicAliases - If the target has any MnemonicAlias<> definitions,
/// emit a function for them and return true, otherwise return false.
static bool emitMnemonicAliases(raw_ostream &OS, const AsmMatcherInfo &Info) {
  // Ignore aliases when match-prefix is set.
  if (!MatchPrefix.empty())
    return false;

  std::vector<Record*> Aliases =
    Info.getRecords().getAllDerivedDefinitions("MnemonicAlias");
  if (Aliases.empty()) return false;

  OS << "static void applyMnemonicAliases(StringRef &Mnemonic, "
        "unsigned Features) {\n";

  // Keep track of all the aliases from a mnemonic.  Use an std::map so that the
  // iteration order of the map is stable.
  std::map<std::string, std::vector<Record*> > AliasesFromMnemonic;

  for (unsigned i = 0, e = Aliases.size(); i != e; ++i) {
    Record *R = Aliases[i];
    AliasesFromMnemonic[R->getValueAsString("FromMnemonic")].push_back(R);
  }

  // Process each alias a "from" mnemonic at a time, building the code executed
  // by the string remapper.
  std::vector<StringMatcher::StringPair> Cases;
  for (std::map<std::string, std::vector<Record*> >::iterator
       I = AliasesFromMnemonic.begin(), E = AliasesFromMnemonic.end();
       I != E; ++I) {
    const std::vector<Record*> &ToVec = I->second;

    // Loop through each alias and emit code that handles each case.  If there
    // are two instructions without predicates, emit an error.  If there is one,
    // emit it last.
    std::string MatchCode;
    int AliasWithNoPredicate = -1;

    for (unsigned i = 0, e = ToVec.size(); i != e; ++i) {
      Record *R = ToVec[i];
      std::string FeatureMask = GetAliasRequiredFeatures(R, Info);

      // If this unconditionally matches, remember it for later and diagnose
      // duplicates.
      if (FeatureMask.empty()) {
        if (AliasWithNoPredicate != -1) {
          // We can't have two aliases from the same mnemonic with no predicate.
          PrintError(ToVec[AliasWithNoPredicate]->getLoc(),
                     "two MnemonicAliases with the same 'from' mnemonic!");
          PrintFatalError(R->getLoc(), "this is the other MnemonicAlias.");
        }

        AliasWithNoPredicate = i;
        continue;
      }
      if (R->getValueAsString("ToMnemonic") == I->first)
        PrintFatalError(R->getLoc(), "MnemonicAlias to the same string");

      if (!MatchCode.empty())
        MatchCode += "else ";
      MatchCode += "if ((Features & " + FeatureMask + ") == "+FeatureMask+")\n";
      MatchCode += "  Mnemonic = \"" +R->getValueAsString("ToMnemonic")+"\";\n";
    }

    if (AliasWithNoPredicate != -1) {
      Record *R = ToVec[AliasWithNoPredicate];
      if (!MatchCode.empty())
        MatchCode += "else\n  ";
      MatchCode += "Mnemonic = \"" + R->getValueAsString("ToMnemonic")+"\";\n";
    }

    MatchCode += "return;";

    Cases.push_back(std::make_pair(I->first, MatchCode));
  }

  StringMatcher("Mnemonic", Cases, OS).Emit();
  OS << "}\n\n";

  return true;
}

static const char *getMinimalTypeForRange(uint64_t Range) {
  assert(Range < 0xFFFFFFFFULL && "Enum too large");
  if (Range > 0xFFFF)
    return "uint32_t";
  if (Range > 0xFF)
    return "uint16_t";
  return "uint8_t";
}

static void emitCustomOperandParsing(raw_ostream &OS, CodeGenTarget &Target,
                              const AsmMatcherInfo &Info, StringRef ClassName,
                              StringToOffsetTable &StringTable,
                              unsigned MaxMnemonicIndex) {
  unsigned MaxMask = 0;
  for (std::vector<OperandMatchEntry>::const_iterator it =
       Info.OperandMatchInfo.begin(), ie = Info.OperandMatchInfo.end();
       it != ie; ++it) {
    MaxMask |= it->OperandMask;
  }

  // Emit the static custom operand parsing table;
  OS << "namespace {\n";
  OS << "  struct OperandMatchEntry {\n";
  OS << "    " << getMinimalTypeForRange(1ULL << Info.SubtargetFeatures.size())
               << " RequiredFeatures;\n";
  OS << "    " << getMinimalTypeForRange(MaxMnemonicIndex)
               << " Mnemonic;\n";
  OS << "    " << getMinimalTypeForRange(Info.Classes.size())
               << " Class;\n";
  OS << "    " << getMinimalTypeForRange(MaxMask)
               << " OperandMask;\n\n";
  OS << "    StringRef getMnemonic() const {\n";
  OS << "      return StringRef(MnemonicTable + Mnemonic + 1,\n";
  OS << "                       MnemonicTable[Mnemonic]);\n";
  OS << "    }\n";
  OS << "  };\n\n";

  OS << "  // Predicate for searching for an opcode.\n";
  OS << "  struct LessOpcodeOperand {\n";
  OS << "    bool operator()(const OperandMatchEntry &LHS, StringRef RHS) {\n";
  OS << "      return LHS.getMnemonic()  < RHS;\n";
  OS << "    }\n";
  OS << "    bool operator()(StringRef LHS, const OperandMatchEntry &RHS) {\n";
  OS << "      return LHS < RHS.getMnemonic();\n";
  OS << "    }\n";
  OS << "    bool operator()(const OperandMatchEntry &LHS,";
  OS << " const OperandMatchEntry &RHS) {\n";
  OS << "      return LHS.getMnemonic() < RHS.getMnemonic();\n";
  OS << "    }\n";
  OS << "  };\n";

  OS << "} // end anonymous namespace.\n\n";

  OS << "static const OperandMatchEntry OperandMatchTable["
     << Info.OperandMatchInfo.size() << "] = {\n";

  OS << "  /* Operand List Mask, Mnemonic, Operand Class, Features */\n";
  for (std::vector<OperandMatchEntry>::const_iterator it =
       Info.OperandMatchInfo.begin(), ie = Info.OperandMatchInfo.end();
       it != ie; ++it) {
    const OperandMatchEntry &OMI = *it;
    const MatchableInfo &II = *OMI.MI;

    OS << "  { ";

    // Write the required features mask.
    if (!II.RequiredFeatures.empty()) {
      for (unsigned i = 0, e = II.RequiredFeatures.size(); i != e; ++i) {
        if (i) OS << "|";
        OS << II.RequiredFeatures[i]->getEnumName();
      }
    } else
      OS << "0";

    // Store a pascal-style length byte in the mnemonic.
    std::string LenMnemonic = char(II.Mnemonic.size()) + II.Mnemonic.str();
    OS << ", " << StringTable.GetOrAddStringOffset(LenMnemonic, false)
       << " /* " << II.Mnemonic << " */, ";

    OS << OMI.CI->Name;

    OS << ", " << OMI.OperandMask;
    OS << " /* ";
    bool printComma = false;
    for (int i = 0, e = 31; i !=e; ++i)
      if (OMI.OperandMask & (1 << i)) {
        if (printComma)
          OS << ", ";
        OS << i;
        printComma = true;
      }
    OS << " */";

    OS << " },\n";
  }
  OS << "};\n\n";

  // Emit the operand class switch to call the correct custom parser for
  // the found operand class.
  OS << Target.getName() << ClassName << "::OperandMatchResultTy "
     << Target.getName() << ClassName << "::\n"
     << "tryCustomParseOperand(SmallVectorImpl<MCParsedAsmOperand*>"
     << " &Operands,\n                      unsigned MCK) {\n\n"
     << "  switch(MCK) {\n";

  for (std::vector<ClassInfo*>::const_iterator it = Info.Classes.begin(),
       ie = Info.Classes.end(); it != ie; ++it) {
    ClassInfo *CI = *it;
    if (CI->ParserMethod.empty())
      continue;
    OS << "  case " << CI->Name << ":\n"
       << "    return " << CI->ParserMethod << "(Operands);\n";
  }

  OS << "  default:\n";
  OS << "    return MatchOperand_NoMatch;\n";
  OS << "  }\n";
  OS << "  return MatchOperand_NoMatch;\n";
  OS << "}\n\n";

  // Emit the static custom operand parser. This code is very similar with
  // the other matcher. Also use MatchResultTy here just in case we go for
  // a better error handling.
  OS << Target.getName() << ClassName << "::OperandMatchResultTy "
     << Target.getName() << ClassName << "::\n"
     << "MatchOperandParserImpl(SmallVectorImpl<MCParsedAsmOperand*>"
     << " &Operands,\n                       StringRef Mnemonic) {\n";

  // Emit code to get the available features.
  OS << "  // Get the current feature set.\n";
  OS << "  unsigned AvailableFeatures = getAvailableFeatures();\n\n";

  OS << "  // Get the next operand index.\n";
  OS << "  unsigned NextOpNum = Operands.size()-1;\n";

  // Emit code to search the table.
  OS << "  // Search the table.\n";
  OS << "  std::pair<const OperandMatchEntry*, const OperandMatchEntry*>";
  OS << " MnemonicRange =\n";
  OS << "    std::equal_range(OperandMatchTable, OperandMatchTable+"
     << Info.OperandMatchInfo.size() << ", Mnemonic,\n"
     << "                     LessOpcodeOperand());\n\n";

  OS << "  if (MnemonicRange.first == MnemonicRange.second)\n";
  OS << "    return MatchOperand_NoMatch;\n\n";

  OS << "  for (const OperandMatchEntry *it = MnemonicRange.first,\n"
     << "       *ie = MnemonicRange.second; it != ie; ++it) {\n";

  OS << "    // equal_range guarantees that instruction mnemonic matches.\n";
  OS << "    assert(Mnemonic == it->getMnemonic());\n\n";

  // Emit check that the required features are available.
  OS << "    // check if the available features match\n";
  OS << "    if ((AvailableFeatures & it->RequiredFeatures) "
     << "!= it->RequiredFeatures) {\n";
  OS << "      continue;\n";
  OS << "    }\n\n";

  // Emit check to ensure the operand number matches.
  OS << "    // check if the operand in question has a custom parser.\n";
  OS << "    if (!(it->OperandMask & (1 << NextOpNum)))\n";
  OS << "      continue;\n\n";

  // Emit call to the custom parser method
  OS << "    // call custom parse method to handle the operand\n";
  OS << "    OperandMatchResultTy Result = ";
  OS << "tryCustomParseOperand(Operands, it->Class);\n";
  OS << "    if (Result != MatchOperand_NoMatch)\n";
  OS << "      return Result;\n";
  OS << "  }\n\n";

  OS << "  // Okay, we had no match.\n";
  OS << "  return MatchOperand_NoMatch;\n";
  OS << "}\n\n";
}

void AsmMatcherEmitter::run(raw_ostream &OS) {
  CodeGenTarget Target(Records);
  Record *AsmParser = Target.getAsmParser();
  std::string ClassName = AsmParser->getValueAsString("AsmParserClassName");

  // Compute the information on the instructions to match.
  AsmMatcherInfo Info(AsmParser, Target, Records);
  Info.buildInfo();

  // Sort the instruction table using the partial order on classes. We use
  // stable_sort to ensure that ambiguous instructions are still
  // deterministically ordered.
  std::stable_sort(Info.Matchables.begin(), Info.Matchables.end(),
                   less_ptr<MatchableInfo>());

  DEBUG_WITH_TYPE("instruction_info", {
      for (std::vector<MatchableInfo*>::iterator
             it = Info.Matchables.begin(), ie = Info.Matchables.end();
           it != ie; ++it)
        (*it)->dump();
    });

  // Check for ambiguous matchables.
  DEBUG_WITH_TYPE("ambiguous_instrs", {
    unsigned NumAmbiguous = 0;
    for (unsigned i = 0, e = Info.Matchables.size(); i != e; ++i) {
      for (unsigned j = i + 1; j != e; ++j) {
        MatchableInfo &A = *Info.Matchables[i];
        MatchableInfo &B = *Info.Matchables[j];

        if (A.couldMatchAmbiguouslyWith(B)) {
          errs() << "warning: ambiguous matchables:\n";
          A.dump();
          errs() << "\nis incomparable with:\n";
          B.dump();
          errs() << "\n\n";
          ++NumAmbiguous;
        }
      }
    }
    if (NumAmbiguous)
      errs() << "warning: " << NumAmbiguous
             << " ambiguous matchables!\n";
  });

  // Compute the information on the custom operand parsing.
  Info.buildOperandMatchInfo();

  // Write the output.

  // Information for the class declaration.
  OS << "\n#ifdef GET_ASSEMBLER_HEADER\n";
  OS << "#undef GET_ASSEMBLER_HEADER\n";
  OS << "  // This should be included into the middle of the declaration of\n";
  OS << "  // your subclasses implementation of MCTargetAsmParser.\n";
  OS << "  unsigned ComputeAvailableFeatures(uint64_t FeatureBits) const;\n";
  OS << "  void convertToMCInst(unsigned Kind, MCInst &Inst, "
     << "unsigned Opcode,\n"
     << "                       const SmallVectorImpl<MCParsedAsmOperand*> "
     << "&Operands);\n";
  OS << "  void convertToMapAndConstraints(unsigned Kind,\n                ";
  OS << "           const SmallVectorImpl<MCParsedAsmOperand*> &Operands);\n";
  OS << "  bool mnemonicIsValid(StringRef Mnemonic);\n";
  OS << "  unsigned MatchInstructionImpl(\n";
  OS.indent(27);
  OS << "const SmallVectorImpl<MCParsedAsmOperand*> &Operands,\n"
     << "                                MCInst &Inst,\n"
     << "                                unsigned &ErrorInfo,"
     << " bool matchingInlineAsm,\n"
     << "                                unsigned VariantID = 0);\n";

  if (Info.OperandMatchInfo.size()) {
    OS << "\n  enum OperandMatchResultTy {\n";
    OS << "    MatchOperand_Success,    // operand matched successfully\n";
    OS << "    MatchOperand_NoMatch,    // operand did not match\n";
    OS << "    MatchOperand_ParseFail   // operand matched but had errors\n";
    OS << "  };\n";
    OS << "  OperandMatchResultTy MatchOperandParserImpl(\n";
    OS << "    SmallVectorImpl<MCParsedAsmOperand*> &Operands,\n";
    OS << "    StringRef Mnemonic);\n";

    OS << "  OperandMatchResultTy tryCustomParseOperand(\n";
    OS << "    SmallVectorImpl<MCParsedAsmOperand*> &Operands,\n";
    OS << "    unsigned MCK);\n\n";
  }

  OS << "#endif // GET_ASSEMBLER_HEADER_INFO\n\n";

  // Emit the operand match diagnostic enum names.
  OS << "\n#ifdef GET_OPERAND_DIAGNOSTIC_TYPES\n";
  OS << "#undef GET_OPERAND_DIAGNOSTIC_TYPES\n\n";
  emitOperandDiagnosticTypes(Info, OS);
  OS << "#endif // GET_OPERAND_DIAGNOSTIC_TYPES\n\n";


  OS << "\n#ifdef GET_REGISTER_MATCHER\n";
  OS << "#undef GET_REGISTER_MATCHER\n\n";

  // Emit the subtarget feature enumeration.
  emitSubtargetFeatureFlagEnumeration(Info, OS);

  // Emit the function to match a register name to number.
  // This should be omitted for Mips target
  if (AsmParser->getValueAsBit("ShouldEmitMatchRegisterName"))
    emitMatchRegisterName(Target, AsmParser, OS);

  OS << "#endif // GET_REGISTER_MATCHER\n\n";

  OS << "\n#ifdef GET_SUBTARGET_FEATURE_NAME\n";
  OS << "#undef GET_SUBTARGET_FEATURE_NAME\n\n";

  // Generate the helper function to get the names for subtarget features.
  emitGetSubtargetFeatureName(Info, OS);

  OS << "#endif // GET_SUBTARGET_FEATURE_NAME\n\n";

  OS << "\n#ifdef GET_MATCHER_IMPLEMENTATION\n";
  OS << "#undef GET_MATCHER_IMPLEMENTATION\n\n";

  // Generate the function that remaps for mnemonic aliases.
  bool HasMnemonicAliases = emitMnemonicAliases(OS, Info);

  // Generate the convertToMCInst function to convert operands into an MCInst.
  // Also, generate the convertToMapAndConstraints function for MS-style inline
  // assembly.  The latter doesn't actually generate a MCInst.
  emitConvertFuncs(Target, ClassName, Info.Matchables, OS);

  // Emit the enumeration for classes which participate in matching.
  emitMatchClassEnumeration(Target, Info.Classes, OS);

  // Emit the routine to match token strings to their match class.
  emitMatchTokenString(Target, Info.Classes, OS);

  // Emit the subclass predicate routine.
  emitIsSubclass(Target, Info.Classes, OS);

  // Emit the routine to validate an operand against a match class.
  emitValidateOperandClass(Info, OS);

  // Emit the available features compute function.
  emitComputeAvailableFeatures(Info, OS);


  StringToOffsetTable StringTable;

  size_t MaxNumOperands = 0;
  unsigned MaxMnemonicIndex = 0;
  for (std::vector<MatchableInfo*>::const_iterator it =
         Info.Matchables.begin(), ie = Info.Matchables.end();
       it != ie; ++it) {
    MatchableInfo &II = **it;
    MaxNumOperands = std::max(MaxNumOperands, II.AsmOperands.size());

    // Store a pascal-style length byte in the mnemonic.
    std::string LenMnemonic = char(II.Mnemonic.size()) + II.Mnemonic.str();
    MaxMnemonicIndex = std::max(MaxMnemonicIndex,
                        StringTable.GetOrAddStringOffset(LenMnemonic, false));
  }

  OS << "static const char *const MnemonicTable =\n";
  StringTable.EmitString(OS);
  OS << ";\n\n";

  // Emit the static match table; unused classes get initalized to 0 which is
  // guaranteed to be InvalidMatchClass.
  //
  // FIXME: We can reduce the size of this table very easily. First, we change
  // it so that store the kinds in separate bit-fields for each index, which
  // only needs to be the max width used for classes at that index (we also need
  // to reject based on this during classification). If we then make sure to
  // order the match kinds appropriately (putting mnemonics last), then we
  // should only end up using a few bits for each class, especially the ones
  // following the mnemonic.
  OS << "namespace {\n";
  OS << "  struct MatchEntry {\n";
  OS << "    " << getMinimalTypeForRange(MaxMnemonicIndex)
               << " Mnemonic;\n";
  OS << "    uint16_t Opcode;\n";
  OS << "    " << getMinimalTypeForRange(Info.Matchables.size())
               << " ConvertFn;\n";
  OS << "    " << getMinimalTypeForRange(1ULL << Info.SubtargetFeatures.size())
               << " RequiredFeatures;\n";
  OS << "    " << getMinimalTypeForRange(Info.Classes.size())
               << " Classes[" << MaxNumOperands << "];\n";
  OS << "    uint8_t AsmVariantID;\n\n";
  OS << "    StringRef getMnemonic() const {\n";
  OS << "      return StringRef(MnemonicTable + Mnemonic + 1,\n";
  OS << "                       MnemonicTable[Mnemonic]);\n";
  OS << "    }\n";
  OS << "  };\n\n";

  OS << "  // Predicate for searching for an opcode.\n";
  OS << "  struct LessOpcode {\n";
  OS << "    bool operator()(const MatchEntry &LHS, StringRef RHS) {\n";
  OS << "      return LHS.getMnemonic() < RHS;\n";
  OS << "    }\n";
  OS << "    bool operator()(StringRef LHS, const MatchEntry &RHS) {\n";
  OS << "      return LHS < RHS.getMnemonic();\n";
  OS << "    }\n";
  OS << "    bool operator()(const MatchEntry &LHS, const MatchEntry &RHS) {\n";
  OS << "      return LHS.getMnemonic() < RHS.getMnemonic();\n";
  OS << "    }\n";
  OS << "  };\n";

  OS << "} // end anonymous namespace.\n\n";

  OS << "static const MatchEntry MatchTable["
     << Info.Matchables.size() << "] = {\n";

  for (std::vector<MatchableInfo*>::const_iterator it =
       Info.Matchables.begin(), ie = Info.Matchables.end();
       it != ie; ++it) {
    MatchableInfo &II = **it;

    // Store a pascal-style length byte in the mnemonic.
    std::string LenMnemonic = char(II.Mnemonic.size()) + II.Mnemonic.str();
    OS << "  { " << StringTable.GetOrAddStringOffset(LenMnemonic, false)
       << " /* " << II.Mnemonic << " */, "
       << Target.getName() << "::"
       << II.getResultInst()->TheDef->getName() << ", "
       << II.ConversionFnKind << ", ";

    // Write the required features mask.
    if (!II.RequiredFeatures.empty()) {
      for (unsigned i = 0, e = II.RequiredFeatures.size(); i != e; ++i) {
        if (i) OS << "|";
        OS << II.RequiredFeatures[i]->getEnumName();
      }
    } else
      OS << "0";

    OS << ", { ";
    for (unsigned i = 0, e = II.AsmOperands.size(); i != e; ++i) {
      MatchableInfo::AsmOperand &Op = II.AsmOperands[i];

      if (i) OS << ", ";
      OS << Op.Class->Name;
    }
    OS << " }, " << II.AsmVariantID;
    OS << "},\n";
  }

  OS << "};\n\n";

  // A method to determine if a mnemonic is in the list.
  OS << "bool " << Target.getName() << ClassName << "::\n"
     << "mnemonicIsValid(StringRef Mnemonic) {\n";
  OS << "  // Search the table.\n";
  OS << "  std::pair<const MatchEntry*, const MatchEntry*> MnemonicRange =\n";
  OS << "    std::equal_range(MatchTable, MatchTable+"
     << Info.Matchables.size() << ", Mnemonic, LessOpcode());\n";
  OS << "  return MnemonicRange.first != MnemonicRange.second;\n";
  OS << "}\n\n";

  // Finally, build the match function.
  OS << "unsigned "
     << Target.getName() << ClassName << "::\n"
     << "MatchInstructionImpl(const SmallVectorImpl<MCParsedAsmOperand*>"
     << " &Operands,\n";
  OS << "                     MCInst &Inst,\n"
     << "unsigned &ErrorInfo, bool matchingInlineAsm, unsigned VariantID) {\n";

  OS << "  // Eliminate obvious mismatches.\n";
  OS << "  if (Operands.size() > " << (MaxNumOperands+1) << ") {\n";
  OS << "    ErrorInfo = " << (MaxNumOperands+1) << ";\n";
  OS << "    return Match_InvalidOperand;\n";
  OS << "  }\n\n";

  // Emit code to get the available features.
  OS << "  // Get the current feature set.\n";
  OS << "  unsigned AvailableFeatures = getAvailableFeatures();\n\n";

  OS << "  // Get the instruction mnemonic, which is the first token.\n";
  OS << "  StringRef Mnemonic = ((" << Target.getName()
     << "Operand*)Operands[0])->getToken();\n\n";

  if (HasMnemonicAliases) {
    OS << "  // Process all MnemonicAliases to remap the mnemonic.\n";
    OS << "  // FIXME : Add an entry in AsmParserVariant to check this.\n";
    OS << "  if (!VariantID)\n";
    OS << "    applyMnemonicAliases(Mnemonic, AvailableFeatures);\n\n";
  }

  // Emit code to compute the class list for this operand vector.
  OS << "  // Some state to try to produce better error messages.\n";
  OS << "  bool HadMatchOtherThanFeatures = false;\n";
  OS << "  bool HadMatchOtherThanPredicate = false;\n";
  OS << "  unsigned RetCode = Match_InvalidOperand;\n";
  OS << "  unsigned MissingFeatures = ~0U;\n";
  OS << "  // Set ErrorInfo to the operand that mismatches if it is\n";
  OS << "  // wrong for all instances of the instruction.\n";
  OS << "  ErrorInfo = ~0U;\n";

  // Emit code to search the table.
  OS << "  // Search the table.\n";
  OS << "  std::pair<const MatchEntry*, const MatchEntry*> MnemonicRange =\n";
  OS << "    std::equal_range(MatchTable, MatchTable+"
     << Info.Matchables.size() << ", Mnemonic, LessOpcode());\n\n";

  OS << "  // Return a more specific error code if no mnemonics match.\n";
  OS << "  if (MnemonicRange.first == MnemonicRange.second)\n";
  OS << "    return Match_MnemonicFail;\n\n";

  OS << "  for (const MatchEntry *it = MnemonicRange.first, "
     << "*ie = MnemonicRange.second;\n";
  OS << "       it != ie; ++it) {\n";

  OS << "    // equal_range guarantees that instruction mnemonic matches.\n";
  OS << "    assert(Mnemonic == it->getMnemonic());\n";

  // Emit check that the subclasses match.
  OS << "    if (VariantID != it->AsmVariantID) continue;\n";
  OS << "    bool OperandsValid = true;\n";
  OS << "    for (unsigned i = 0; i != " << MaxNumOperands << "; ++i) {\n";
  OS << "      if (i + 1 >= Operands.size()) {\n";
  OS << "        OperandsValid = (it->Classes[i] == " <<"InvalidMatchClass);\n";
  OS << "        if (!OperandsValid) ErrorInfo = i + 1;\n";
  OS << "        break;\n";
  OS << "      }\n";
  OS << "      unsigned Diag = validateOperandClass(Operands[i+1],\n";
  OS.indent(43);
  OS << "(MatchClassKind)it->Classes[i]);\n";
  OS << "      if (Diag == Match_Success)\n";
  OS << "        continue;\n";
  OS << "      // If this operand is broken for all of the instances of this\n";
  OS << "      // mnemonic, keep track of it so we can report loc info.\n";
  OS << "      // If we already had a match that only failed due to a\n";
  OS << "      // target predicate, that diagnostic is preferred.\n";
  OS << "      if (!HadMatchOtherThanPredicate &&\n";
  OS << "          (it == MnemonicRange.first || ErrorInfo <= i+1)) {\n";
  OS << "        ErrorInfo = i+1;\n";
  OS << "        // InvalidOperand is the default. Prefer specificity.\n";
  OS << "        if (Diag != Match_InvalidOperand)\n";
  OS << "          RetCode = Diag;\n";
  OS << "      }\n";
  OS << "      // Otherwise, just reject this instance of the mnemonic.\n";
  OS << "      OperandsValid = false;\n";
  OS << "      break;\n";
  OS << "    }\n\n";

  OS << "    if (!OperandsValid) continue;\n";

  // Emit check that the required features are available.
  OS << "    if ((AvailableFeatures & it->RequiredFeatures) "
     << "!= it->RequiredFeatures) {\n";
  OS << "      HadMatchOtherThanFeatures = true;\n";
  OS << "      unsigned NewMissingFeatures = it->RequiredFeatures & "
        "~AvailableFeatures;\n";
  OS << "      if (CountPopulation_32(NewMissingFeatures) <=\n"
        "          CountPopulation_32(MissingFeatures))\n";
  OS << "        MissingFeatures = NewMissingFeatures;\n";
  OS << "      continue;\n";
  OS << "    }\n";
  OS << "\n";
  OS << "    if (matchingInlineAsm) {\n";
  OS << "      Inst.setOpcode(it->Opcode);\n";
  OS << "      convertToMapAndConstraints(it->ConvertFn, Operands);\n";
  OS << "      return Match_Success;\n";
  OS << "    }\n\n";
  OS << "    // We have selected a definite instruction, convert the parsed\n"
     << "    // operands into the appropriate MCInst.\n";
  OS << "    convertToMCInst(it->ConvertFn, Inst, it->Opcode, Operands);\n";
  OS << "\n";

  // Verify the instruction with the target-specific match predicate function.
  OS << "    // We have a potential match. Check the target predicate to\n"
     << "    // handle any context sensitive constraints.\n"
     << "    unsigned MatchResult;\n"
     << "    if ((MatchResult = checkTargetMatchPredicate(Inst)) !="
     << " Match_Success) {\n"
     << "      Inst.clear();\n"
     << "      RetCode = MatchResult;\n"
     << "      HadMatchOtherThanPredicate = true;\n"
     << "      continue;\n"
     << "    }\n\n";

  // Call the post-processing function, if used.
  std::string InsnCleanupFn =
    AsmParser->getValueAsString("AsmParserInstCleanup");
  if (!InsnCleanupFn.empty())
    OS << "    " << InsnCleanupFn << "(Inst);\n";

  OS << "    return Match_Success;\n";
  OS << "  }\n\n";

  OS << "  // Okay, we had no match.  Try to return a useful error code.\n";
  OS << "  if (HadMatchOtherThanPredicate || !HadMatchOtherThanFeatures)\n";
  OS << "    return RetCode;\n\n";
  OS << "  // Missing feature matches return which features were missing\n";
  OS << "  ErrorInfo = MissingFeatures;\n";
  OS << "  return Match_MissingFeature;\n";
  OS << "}\n\n";

  if (Info.OperandMatchInfo.size())
    emitCustomOperandParsing(OS, Target, Info, ClassName, StringTable,
                             MaxMnemonicIndex);

  OS << "#endif // GET_MATCHER_IMPLEMENTATION\n\n";
}

namespace llvm {

void EmitAsmMatcher(RecordKeeper &RK, raw_ostream &OS) {
  emitSourceFileHeader("Assembly Matcher Source Fragment", OS);
  AsmMatcherEmitter(RK).run(OS);
}

} // End llvm namespace
