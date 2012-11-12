//===- X86DisassemblerTables.cpp - Disassembler tables ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is part of the X86 Disassembler Emitter.
// It contains the implementation of the disassembler tables.
// Documentation for the disassembler emitter in general can be found in
//  X86DisasemblerEmitter.h.
//
//===----------------------------------------------------------------------===//

#include "X86DisassemblerShared.h"
#include "X86DisassemblerTables.h"

#include "llvm/TableGen/TableGenBackend.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Format.h"
#include <map>

using namespace llvm;
using namespace X86Disassembler;

/// inheritsFrom - Indicates whether all instructions in one class also belong
///   to another class.
///
/// @param child  - The class that may be the subset
/// @param parent - The class that may be the superset
/// @return       - True if child is a subset of parent, false otherwise.
static inline bool inheritsFrom(InstructionContext child,
                                InstructionContext parent,
                                bool VEX_LIG = false) {
  if (child == parent)
    return true;

  switch (parent) {
  case IC:
    return(inheritsFrom(child, IC_64BIT) ||
           inheritsFrom(child, IC_OPSIZE) ||
           inheritsFrom(child, IC_ADSIZE) ||
           inheritsFrom(child, IC_XD) ||
           inheritsFrom(child, IC_XS));
  case IC_64BIT:
    return(inheritsFrom(child, IC_64BIT_REXW)   ||
           inheritsFrom(child, IC_64BIT_OPSIZE) ||
           inheritsFrom(child, IC_64BIT_ADSIZE) ||
           inheritsFrom(child, IC_64BIT_XD)     ||
           inheritsFrom(child, IC_64BIT_XS));
  case IC_OPSIZE:
    return inheritsFrom(child, IC_64BIT_OPSIZE);
  case IC_ADSIZE:
  case IC_64BIT_ADSIZE:
    return false;
  case IC_XD:
    return inheritsFrom(child, IC_64BIT_XD);
  case IC_XS:
    return inheritsFrom(child, IC_64BIT_XS);
  case IC_XD_OPSIZE:
    return inheritsFrom(child, IC_64BIT_XD_OPSIZE);
  case IC_XS_OPSIZE:
    return inheritsFrom(child, IC_64BIT_XS_OPSIZE);
  case IC_64BIT_REXW:
    return(inheritsFrom(child, IC_64BIT_REXW_XS) ||
           inheritsFrom(child, IC_64BIT_REXW_XD) ||
           inheritsFrom(child, IC_64BIT_REXW_OPSIZE));
  case IC_64BIT_OPSIZE:
    return(inheritsFrom(child, IC_64BIT_REXW_OPSIZE));
  case IC_64BIT_XD:
    return(inheritsFrom(child, IC_64BIT_REXW_XD));
  case IC_64BIT_XS:
    return(inheritsFrom(child, IC_64BIT_REXW_XS));
  case IC_64BIT_XD_OPSIZE:
  case IC_64BIT_XS_OPSIZE:
    return false;
  case IC_64BIT_REXW_XD:
  case IC_64BIT_REXW_XS:
  case IC_64BIT_REXW_OPSIZE:
    return false;
  case IC_VEX:
    return inheritsFrom(child, IC_VEX_W) ||
           (VEX_LIG && inheritsFrom(child, IC_VEX_L));
  case IC_VEX_XS:
    return inheritsFrom(child, IC_VEX_W_XS) ||
           (VEX_LIG && inheritsFrom(child, IC_VEX_L_XS));
  case IC_VEX_XD:
    return inheritsFrom(child, IC_VEX_W_XD) ||
           (VEX_LIG && inheritsFrom(child, IC_VEX_L_XD));
  case IC_VEX_OPSIZE:
    return inheritsFrom(child, IC_VEX_W_OPSIZE) ||
           (VEX_LIG && inheritsFrom(child, IC_VEX_L_OPSIZE));
  case IC_VEX_W:
  case IC_VEX_W_XS:
  case IC_VEX_W_XD:
  case IC_VEX_W_OPSIZE:
    return false;
  case IC_VEX_L:
  case IC_VEX_L_XS:
  case IC_VEX_L_XD:
    return false;
  case IC_VEX_L_OPSIZE:
    return inheritsFrom(child, IC_VEX_L_W_OPSIZE);
  case IC_VEX_L_W_OPSIZE:
    return false;
  default:
    llvm_unreachable("Unknown instruction class");
  }
}

/// outranks - Indicates whether, if an instruction has two different applicable
///   classes, which class should be preferred when performing decode.  This
///   imposes a total ordering (ties are resolved toward "lower")
///
/// @param upper  - The class that may be preferable
/// @param lower  - The class that may be less preferable
/// @return       - True if upper is to be preferred, false otherwise.
static inline bool outranks(InstructionContext upper,
                            InstructionContext lower) {
  assert(upper < IC_max);
  assert(lower < IC_max);

#define ENUM_ENTRY(n, r, d) r,
  static int ranks[IC_max] = {
    INSTRUCTION_CONTEXTS
  };
#undef ENUM_ENTRY

  return (ranks[upper] > ranks[lower]);
}

/// stringForContext - Returns a string containing the name of a particular
///   InstructionContext, usually for diagnostic purposes.
///
/// @param insnContext  - The instruction class to transform to a string.
/// @return           - A statically-allocated string constant that contains the
///                     name of the instruction class.
static inline const char* stringForContext(InstructionContext insnContext) {
  switch (insnContext) {
  default:
    llvm_unreachable("Unhandled instruction class");
#define ENUM_ENTRY(n, r, d)   case n: return #n; break;
  INSTRUCTION_CONTEXTS
#undef ENUM_ENTRY
  }
}

/// stringForOperandType - Like stringForContext, but for OperandTypes.
static inline const char* stringForOperandType(OperandType type) {
  switch (type) {
  default:
    llvm_unreachable("Unhandled type");
#define ENUM_ENTRY(i, d) case i: return #i;
  TYPES
#undef ENUM_ENTRY
  }
}

/// stringForOperandEncoding - like stringForContext, but for
///   OperandEncodings.
static inline const char* stringForOperandEncoding(OperandEncoding encoding) {
  switch (encoding) {
  default:
    llvm_unreachable("Unhandled encoding");
#define ENUM_ENTRY(i, d) case i: return #i;
  ENCODINGS
#undef ENUM_ENTRY
  }
}

void DisassemblerTables::emitOneID(raw_ostream &o, unsigned &i, InstrUID id,
                                   bool addComma) const {
  if (id)
    o.indent(i * 2) << format("0x%hx", id);
  else
    o.indent(i * 2) << 0;

  if (addComma)
    o << ", ";
  else
    o << "  ";

  o << "/* ";
  o << InstructionSpecifiers[id].name;
  o << "*/";

  o << "\n";
}

/// emitEmptyTable - Emits the modRMEmptyTable, which is used as a ID table by
///   all ModR/M decisions for instructions that are invalid for all possible
///   ModR/M byte values.
///
/// @param o        - The output stream on which to emit the table.
/// @param i        - The indentation level for that output stream.
static void emitEmptyTable(raw_ostream &o, unsigned &i) {
  o.indent(i * 2) << "0x0, /* EmptyTable */\n";
}

/// getDecisionType - Determines whether a ModRM decision with 255 entries can
///   be compacted by eliminating redundant information.
///
/// @param decision - The decision to be compacted.
/// @return         - The compactest available representation for the decision.
static ModRMDecisionType getDecisionType(ModRMDecision &decision) {
  bool satisfiesOneEntry = true;
  bool satisfiesSplitRM = true;
  bool satisfiesSplitReg = true;
  bool satisfiesSplitMisc = true;

  for (unsigned index = 0; index < 256; ++index) {
    if (decision.instructionIDs[index] != decision.instructionIDs[0])
      satisfiesOneEntry = false;

    if (((index & 0xc0) == 0xc0) &&
       (decision.instructionIDs[index] != decision.instructionIDs[0xc0]))
      satisfiesSplitRM = false;

    if (((index & 0xc0) != 0xc0) &&
       (decision.instructionIDs[index] != decision.instructionIDs[0x00]))
      satisfiesSplitRM = false;

    if (((index & 0xc0) == 0xc0) &&
       (decision.instructionIDs[index] != decision.instructionIDs[index&0xf8]))
      satisfiesSplitReg = false;

    if (((index & 0xc0) != 0xc0) &&
       (decision.instructionIDs[index] != decision.instructionIDs[index&0x38]))
      satisfiesSplitMisc = false;
  }

  if (satisfiesOneEntry)
    return MODRM_ONEENTRY;

  if (satisfiesSplitRM)
    return MODRM_SPLITRM;

  if (satisfiesSplitReg && satisfiesSplitMisc)
    return MODRM_SPLITREG;

  if (satisfiesSplitMisc)
    return MODRM_SPLITMISC;

  return MODRM_FULL;
}

/// stringForDecisionType - Returns a statically-allocated string corresponding
///   to a particular decision type.
///
/// @param dt - The decision type.
/// @return   - A pointer to the statically-allocated string (e.g.,
///             "MODRM_ONEENTRY" for MODRM_ONEENTRY).
static const char* stringForDecisionType(ModRMDecisionType dt) {
#define ENUM_ENTRY(n) case n: return #n;
  switch (dt) {
    default:
      llvm_unreachable("Unknown decision type");
    MODRMTYPES
  };
#undef ENUM_ENTRY
}

/// stringForModifierType - Returns a statically-allocated string corresponding
///   to an opcode modifier type.
///
/// @param mt - The modifier type.
/// @return   - A pointer to the statically-allocated string (e.g.,
///             "MODIFIER_NONE" for MODIFIER_NONE).
static const char* stringForModifierType(ModifierType mt) {
#define ENUM_ENTRY(n) case n: return #n;
  switch(mt) {
    default:
      llvm_unreachable("Unknown modifier type");
    MODIFIER_TYPES
  };
#undef ENUM_ENTRY
}

DisassemblerTables::DisassemblerTables() {
  unsigned i;

  for (i = 0; i < array_lengthof(Tables); i++) {
    Tables[i] = new ContextDecision;
    memset(Tables[i], 0, sizeof(ContextDecision));
  }

  HasConflicts = false;
}

DisassemblerTables::~DisassemblerTables() {
  unsigned i;

  for (i = 0; i < array_lengthof(Tables); i++)
    delete Tables[i];
}

void DisassemblerTables::emitModRMDecision(raw_ostream &o1, raw_ostream &o2,
                                           unsigned &i1, unsigned &i2,
                                           ModRMDecision &decision) const {
  static uint32_t sTableNumber = 0;
  static uint32_t sEntryNumber = 1;
  ModRMDecisionType dt = getDecisionType(decision);

  if (dt == MODRM_ONEENTRY && decision.instructionIDs[0] == 0)
  {
    o2.indent(i2) << "{ /* ModRMDecision */" << "\n";
    i2++;

    o2.indent(i2) << stringForDecisionType(dt) << "," << "\n";
    o2.indent(i2) << 0 << " /* EmptyTable */\n";

    i2--;
    o2.indent(i2) << "}";
    return;
  }

  o1 << "/* Table" << sTableNumber << " */\n";
  i1++;

  switch (dt) {
    default:
      llvm_unreachable("Unknown decision type");
    case MODRM_ONEENTRY:
      emitOneID(o1, i1, decision.instructionIDs[0], true);
      break;
    case MODRM_SPLITRM:
      emitOneID(o1, i1, decision.instructionIDs[0x00], true); // mod = 0b00
      emitOneID(o1, i1, decision.instructionIDs[0xc0], true); // mod = 0b11
      break;
    case MODRM_SPLITREG:
      for (unsigned index = 0; index < 64; index += 8)
        emitOneID(o1, i1, decision.instructionIDs[index], true);
      for (unsigned index = 0xc0; index < 256; index += 8)
        emitOneID(o1, i1, decision.instructionIDs[index], true);
      break;
    case MODRM_SPLITMISC:
      for (unsigned index = 0; index < 64; index += 8)
        emitOneID(o1, i1, decision.instructionIDs[index], true);
      for (unsigned index = 0xc0; index < 256; ++index)
        emitOneID(o1, i1, decision.instructionIDs[index], true);
      break;
    case MODRM_FULL:
      for (unsigned index = 0; index < 256; ++index)
        emitOneID(o1, i1, decision.instructionIDs[index], true);
      break;
  }

  i1--;

  o2.indent(i2) << "{ /* struct ModRMDecision */" << "\n";
  i2++;

  o2.indent(i2) << stringForDecisionType(dt) << "," << "\n";
  o2.indent(i2) << sEntryNumber << " /* Table" << sTableNumber << " */\n";

  i2--;
  o2.indent(i2) << "}";

  switch (dt) {
    default:
      llvm_unreachable("Unknown decision type");
    case MODRM_ONEENTRY:
      sEntryNumber += 1;
      break;
    case MODRM_SPLITRM:
      sEntryNumber += 2;
      break;
    case MODRM_SPLITREG:
      sEntryNumber += 16;
      break;
    case MODRM_SPLITMISC:
      sEntryNumber += 8 + 64;
      break;
    case MODRM_FULL:
      sEntryNumber += 256;
      break;
  }

  // We assume that the index can fit into uint16_t.
  assert(sEntryNumber < 65536U &&
         "Index into ModRMDecision is too large for uint16_t!");

  ++sTableNumber;
}

void DisassemblerTables::emitOpcodeDecision(raw_ostream &o1, raw_ostream &o2,
                                            unsigned &i1, unsigned &i2,
                                            OpcodeDecision &decision) const {
  o2.indent(i2) << "{ /* struct OpcodeDecision */" << "\n";
  i2++;
  o2.indent(i2) << "{" << "\n";
  i2++;

  for (unsigned index = 0; index < 256; ++index) {
    o2.indent(i2);

    o2 << "/* 0x" << format("%02hhx", index) << " */" << "\n";

    emitModRMDecision(o1, o2, i1, i2, decision.modRMDecisions[index]);

    if (index <  255)
      o2 << ",";

    o2 << "\n";
  }

  i2--;
  o2.indent(i2) << "}" << "\n";
  i2--;
  o2.indent(i2) << "}" << "\n";
}

void DisassemblerTables::emitContextDecision(raw_ostream &o1, raw_ostream &o2,
                                             unsigned &i1, unsigned &i2,
                                             ContextDecision &decision,
                                             const char* name) const {
  o2.indent(i2) << "static const struct ContextDecision " << name << " = {\n";
  i2++;
  o2.indent(i2) << "{ /* opcodeDecisions */" << "\n";
  i2++;

  for (unsigned index = 0; index < IC_max; ++index) {
    o2.indent(i2) << "/* ";
    o2 << stringForContext((InstructionContext)index);
    o2 << " */";
    o2 << "\n";

    emitOpcodeDecision(o1, o2, i1, i2, decision.opcodeDecisions[index]);

    if (index + 1 < IC_max)
      o2 << ", ";
  }

  i2--;
  o2.indent(i2) << "}" << "\n";
  i2--;
  o2.indent(i2) << "};" << "\n";
}

void DisassemblerTables::emitInstructionInfo(raw_ostream &o,
                                             unsigned &i) const {
  unsigned NumInstructions = InstructionSpecifiers.size();

  o << "static const struct OperandSpecifier x86OperandSets[]["
    << X86_MAX_OPERANDS << "] = {\n";

  typedef std::vector<std::pair<const char *, const char *> > OperandListTy;
  std::map<OperandListTy, unsigned> OperandSets;

  unsigned OperandSetNum = 0;
  for (unsigned Index = 0; Index < NumInstructions; ++Index) {
    OperandListTy OperandList;

    for (unsigned OperandIndex = 0; OperandIndex < X86_MAX_OPERANDS;
         ++OperandIndex) {
      const char *Encoding =
        stringForOperandEncoding((OperandEncoding)InstructionSpecifiers[Index]
                                 .operands[OperandIndex].encoding);
      const char *Type =
        stringForOperandType((OperandType)InstructionSpecifiers[Index]
                             .operands[OperandIndex].type);
      OperandList.push_back(std::make_pair(Encoding, Type));
    }
    unsigned &N = OperandSets[OperandList];
    if (N != 0) continue;

    N = ++OperandSetNum;

    o << "  { /* " << (OperandSetNum - 1) << " */\n";
    for (unsigned i = 0, e = OperandList.size(); i != e; ++i) {
      o << "    { " << OperandList[i].first << ", "
        << OperandList[i].second << " },\n";
    }
    o << "  },\n";
  }
  o << "};" << "\n\n";

  o.indent(i * 2) << "static const struct InstructionSpecifier ";
  o << INSTRUCTIONS_STR "[" << InstructionSpecifiers.size() << "] = {\n";

  i++;

  for (unsigned index = 0; index < NumInstructions; ++index) {
    o.indent(i * 2) << "{ /* " << index << " */" << "\n";
    i++;

    o.indent(i * 2) << stringForModifierType(
                       (ModifierType)InstructionSpecifiers[index].modifierType);
    o << ",\n";

    o.indent(i * 2) << "0x";
    o << format("%02hhx", (uint16_t)InstructionSpecifiers[index].modifierBase);
    o << ",\n";

    OperandListTy OperandList;
    for (unsigned OperandIndex = 0; OperandIndex < X86_MAX_OPERANDS;
         ++OperandIndex) {
      const char *Encoding =
        stringForOperandEncoding((OperandEncoding)InstructionSpecifiers[index]
                                 .operands[OperandIndex].encoding);
      const char *Type =
        stringForOperandType((OperandType)InstructionSpecifiers[index]
                             .operands[OperandIndex].type);
      OperandList.push_back(std::make_pair(Encoding, Type));
    }
    o.indent(i * 2) << (OperandSets[OperandList] - 1) << ",\n";

    o.indent(i * 2) << "/* " << InstructionSpecifiers[index].name << " */";
    o << "\n";

    i--;
    o.indent(i * 2) << "}";

    if (index + 1 < NumInstructions)
      o << ",";

    o << "\n";
  }

  i--;
  o.indent(i * 2) << "};" << "\n";
}

void DisassemblerTables::emitContextTable(raw_ostream &o, unsigned &i) const {
  o.indent(i * 2) << "static const uint8_t " CONTEXTS_STR
                     "[256] = {\n";
  i++;

  for (unsigned index = 0; index < 256; ++index) {
    o.indent(i * 2);

    if ((index & ATTR_VEXL) && (index & ATTR_REXW) && (index & ATTR_OPSIZE))
      o << "IC_VEX_L_W_OPSIZE";
    else if ((index & ATTR_VEXL) && (index & ATTR_OPSIZE))
      o << "IC_VEX_L_OPSIZE";
    else if ((index & ATTR_VEXL) && (index & ATTR_XD))
      o << "IC_VEX_L_XD";
    else if ((index & ATTR_VEXL) && (index & ATTR_XS))
      o << "IC_VEX_L_XS";
    else if ((index & ATTR_VEX) && (index & ATTR_REXW) && (index & ATTR_OPSIZE))
      o << "IC_VEX_W_OPSIZE";
    else if ((index & ATTR_VEX) && (index & ATTR_REXW) && (index & ATTR_XD))
      o << "IC_VEX_W_XD";
    else if ((index & ATTR_VEX) && (index & ATTR_REXW) && (index & ATTR_XS))
      o << "IC_VEX_W_XS";
    else if (index & ATTR_VEXL)
      o << "IC_VEX_L";
    else if ((index & ATTR_VEX) && (index & ATTR_REXW))
      o << "IC_VEX_W";
    else if ((index & ATTR_VEX) && (index & ATTR_OPSIZE))
      o << "IC_VEX_OPSIZE";
    else if ((index & ATTR_VEX) && (index & ATTR_XD))
      o << "IC_VEX_XD";
    else if ((index & ATTR_VEX) && (index & ATTR_XS))
      o << "IC_VEX_XS";
    else if (index & ATTR_VEX)
      o << "IC_VEX";
    else if ((index & ATTR_64BIT) && (index & ATTR_REXW) && (index & ATTR_XS))
      o << "IC_64BIT_REXW_XS";
    else if ((index & ATTR_64BIT) && (index & ATTR_REXW) && (index & ATTR_XD))
      o << "IC_64BIT_REXW_XD";
    else if ((index & ATTR_64BIT) && (index & ATTR_REXW) &&
             (index & ATTR_OPSIZE))
      o << "IC_64BIT_REXW_OPSIZE";
    else if ((index & ATTR_64BIT) && (index & ATTR_XD) && (index & ATTR_OPSIZE))
      o << "IC_64BIT_XD_OPSIZE";
    else if ((index & ATTR_64BIT) && (index & ATTR_XS) && (index & ATTR_OPSIZE))
      o << "IC_64BIT_XS_OPSIZE";
    else if ((index & ATTR_64BIT) && (index & ATTR_XS))
      o << "IC_64BIT_XS";
    else if ((index & ATTR_64BIT) && (index & ATTR_XD))
      o << "IC_64BIT_XD";
    else if ((index & ATTR_64BIT) && (index & ATTR_OPSIZE))
      o << "IC_64BIT_OPSIZE";
    else if ((index & ATTR_64BIT) && (index & ATTR_ADSIZE))
      o << "IC_64BIT_ADSIZE";
    else if ((index & ATTR_64BIT) && (index & ATTR_REXW))
      o << "IC_64BIT_REXW";
    else if ((index & ATTR_64BIT))
      o << "IC_64BIT";
    else if ((index & ATTR_XS) && (index & ATTR_OPSIZE))
      o << "IC_XS_OPSIZE";
    else if ((index & ATTR_XD) && (index & ATTR_OPSIZE))
      o << "IC_XD_OPSIZE";
    else if (index & ATTR_XS)
      o << "IC_XS";
    else if (index & ATTR_XD)
      o << "IC_XD";
    else if (index & ATTR_OPSIZE)
      o << "IC_OPSIZE";
    else if (index & ATTR_ADSIZE)
      o << "IC_ADSIZE";
    else
      o << "IC";

    if (index < 255)
      o << ",";
    else
      o << " ";

    o << " /* " << index << " */";

    o << "\n";
  }

  i--;
  o.indent(i * 2) << "};" << "\n";
}

void DisassemblerTables::emitContextDecisions(raw_ostream &o1, raw_ostream &o2,
                                             unsigned &i1, unsigned &i2) const {
  emitContextDecision(o1, o2, i1, i2, *Tables[0], ONEBYTE_STR);
  emitContextDecision(o1, o2, i1, i2, *Tables[1], TWOBYTE_STR);
  emitContextDecision(o1, o2, i1, i2, *Tables[2], THREEBYTE38_STR);
  emitContextDecision(o1, o2, i1, i2, *Tables[3], THREEBYTE3A_STR);
  emitContextDecision(o1, o2, i1, i2, *Tables[4], THREEBYTEA6_STR);
  emitContextDecision(o1, o2, i1, i2, *Tables[5], THREEBYTEA7_STR);
}

void DisassemblerTables::emit(raw_ostream &o) const {
  unsigned i1 = 0;
  unsigned i2 = 0;

  std::string s1;
  std::string s2;

  raw_string_ostream o1(s1);
  raw_string_ostream o2(s2);

  emitInstructionInfo(o, i2);
  o << "\n";

  emitContextTable(o, i2);
  o << "\n";

  o << "static const InstrUID modRMTable[] = {\n";
  i1++;
  emitEmptyTable(o1, i1);
  i1--;
  emitContextDecisions(o1, o2, i1, i2);

  o << o1.str();
  o << "  0x0\n";
  o << "};\n";
  o << "\n";
  o << o2.str();
  o << "\n";
  o << "\n";
}

void DisassemblerTables::setTableFields(ModRMDecision     &decision,
                                        const ModRMFilter &filter,
                                        InstrUID          uid,
                                        uint8_t           opcode) {
  for (unsigned index = 0; index < 256; ++index) {
    if (filter.accepts(index)) {
      if (decision.instructionIDs[index] == uid)
        continue;

      if (decision.instructionIDs[index] != 0) {
        InstructionSpecifier &newInfo =
          InstructionSpecifiers[uid];
        InstructionSpecifier &previousInfo =
          InstructionSpecifiers[decision.instructionIDs[index]];

        if(newInfo.filtered)
          continue; // filtered instructions get lowest priority

        if(previousInfo.name == "NOOP" && (newInfo.name == "XCHG16ar" ||
                                           newInfo.name == "XCHG32ar" ||
                                           newInfo.name == "XCHG32ar64" ||
                                           newInfo.name == "XCHG64ar"))
          continue; // special case for XCHG*ar and NOOP

        if (outranks(previousInfo.insnContext, newInfo.insnContext))
          continue;

        if (previousInfo.insnContext == newInfo.insnContext &&
            !previousInfo.filtered) {
          errs() << "Error: Primary decode conflict: ";
          errs() << newInfo.name << " would overwrite " << previousInfo.name;
          errs() << "\n";
          errs() << "ModRM   " << index << "\n";
          errs() << "Opcode  " << (uint16_t)opcode << "\n";
          errs() << "Context " << stringForContext(newInfo.insnContext) << "\n";
          HasConflicts = true;
        }
      }

      decision.instructionIDs[index] = uid;
    }
  }
}

void DisassemblerTables::setTableFields(OpcodeType          type,
                                        InstructionContext  insnContext,
                                        uint8_t             opcode,
                                        const ModRMFilter   &filter,
                                        InstrUID            uid,
                                        bool                is32bit,
                                        bool                ignoresVEX_L) {
  ContextDecision &decision = *Tables[type];

  for (unsigned index = 0; index < IC_max; ++index) {
    if (is32bit && inheritsFrom((InstructionContext)index, IC_64BIT))
      continue;

    if (inheritsFrom((InstructionContext)index,
                     InstructionSpecifiers[uid].insnContext, ignoresVEX_L))
      setTableFields(decision.opcodeDecisions[index].modRMDecisions[opcode],
                     filter,
                     uid,
                     opcode);
  }
}
