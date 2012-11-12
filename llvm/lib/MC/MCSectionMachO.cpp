//===- lib/MC/MCSectionMachO.cpp - MachO Code Section Representation ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCSectionMachO.h"
#include "llvm/MC/MCContext.h"
#include "llvm/Support/raw_ostream.h"
#include <cctype>
using namespace llvm;

/// SectionTypeDescriptors - These are strings that describe the various section
/// types.  This *must* be kept in order with and stay synchronized with the
/// section type list.
static const struct {
  const char *AssemblerName, *EnumName;
} SectionTypeDescriptors[MCSectionMachO::LAST_KNOWN_SECTION_TYPE+1] = {
  { "regular",                  "S_REGULAR" },                    // 0x00
  { 0,                          "S_ZEROFILL" },                   // 0x01
  { "cstring_literals",         "S_CSTRING_LITERALS" },           // 0x02
  { "4byte_literals",           "S_4BYTE_LITERALS" },             // 0x03
  { "8byte_literals",           "S_8BYTE_LITERALS" },             // 0x04
  { "literal_pointers",         "S_LITERAL_POINTERS" },           // 0x05
  { "non_lazy_symbol_pointers", "S_NON_LAZY_SYMBOL_POINTERS" },   // 0x06
  { "lazy_symbol_pointers",     "S_LAZY_SYMBOL_POINTERS" },       // 0x07
  { "symbol_stubs",             "S_SYMBOL_STUBS" },               // 0x08
  { "mod_init_funcs",           "S_MOD_INIT_FUNC_POINTERS" },     // 0x09
  { "mod_term_funcs",           "S_MOD_TERM_FUNC_POINTERS" },     // 0x0A
  { "coalesced",                "S_COALESCED" },                  // 0x0B
  { 0, /*FIXME??*/              "S_GB_ZEROFILL" },                // 0x0C
  { "interposing",              "S_INTERPOSING" },                // 0x0D
  { "16byte_literals",          "S_16BYTE_LITERALS" },            // 0x0E
  { 0, /*FIXME??*/              "S_DTRACE_DOF" },                 // 0x0F
  { 0, /*FIXME??*/              "S_LAZY_DYLIB_SYMBOL_POINTERS" }, // 0x10
  { "thread_local_regular",     "S_THREAD_LOCAL_REGULAR" },       // 0x11
  { "thread_local_zerofill",    "S_THREAD_LOCAL_ZEROFILL" },      // 0x12
  { "thread_local_variables",   "S_THREAD_LOCAL_VARIABLES" },     // 0x13
  { "thread_local_variable_pointers",
    "S_THREAD_LOCAL_VARIABLE_POINTERS" },                         // 0x14
  { "thread_local_init_function_pointers",
    "S_THREAD_LOCAL_INIT_FUNCTION_POINTERS"},                     // 0x15
};


/// SectionAttrDescriptors - This is an array of descriptors for section
/// attributes.  Unlike the SectionTypeDescriptors, this is not directly indexed
/// by attribute, instead it is searched.  The last entry has an AttrFlagEnd
/// AttrFlag value.
static const struct {
  unsigned AttrFlag;
  const char *AssemblerName, *EnumName;
} SectionAttrDescriptors[] = {
#define ENTRY(ASMNAME, ENUM) \
  { MCSectionMachO::ENUM, ASMNAME, #ENUM },
ENTRY("pure_instructions",   S_ATTR_PURE_INSTRUCTIONS)
ENTRY("no_toc",              S_ATTR_NO_TOC)
ENTRY("strip_static_syms",   S_ATTR_STRIP_STATIC_SYMS)
ENTRY("no_dead_strip",       S_ATTR_NO_DEAD_STRIP)
ENTRY("live_support",        S_ATTR_LIVE_SUPPORT)
ENTRY("self_modifying_code", S_ATTR_SELF_MODIFYING_CODE)
ENTRY("debug",               S_ATTR_DEBUG)
ENTRY(0 /*FIXME*/,           S_ATTR_SOME_INSTRUCTIONS)
ENTRY(0 /*FIXME*/,           S_ATTR_EXT_RELOC)
ENTRY(0 /*FIXME*/,           S_ATTR_LOC_RELOC)
#undef ENTRY
  { 0, "none", 0 }, // used if section has no attributes but has a stub size
#define AttrFlagEnd 0xffffffff // non legal value, multiple attribute bits set
  { AttrFlagEnd, 0, 0 }
};

MCSectionMachO::MCSectionMachO(StringRef Segment, StringRef Section,
                               unsigned TAA, unsigned reserved2, SectionKind K)
  : MCSection(SV_MachO, K), TypeAndAttributes(TAA), Reserved2(reserved2) {
  assert(Segment.size() <= 16 && Section.size() <= 16 &&
         "Segment or section string too long");
  for (unsigned i = 0; i != 16; ++i) {
    if (i < Segment.size())
      SegmentName[i] = Segment[i];
    else
      SegmentName[i] = 0;

    if (i < Section.size())
      SectionName[i] = Section[i];
    else
      SectionName[i] = 0;
  }
}

void MCSectionMachO::PrintSwitchToSection(const MCAsmInfo &MAI,
                                          raw_ostream &OS) const {
  OS << "\t.section\t" << getSegmentName() << ',' << getSectionName();

  // Get the section type and attributes.
  unsigned TAA = getTypeAndAttributes();
  if (TAA == 0) {
    OS << '\n';
    return;
  }

  unsigned SectionType = TAA & MCSectionMachO::SECTION_TYPE;
  assert(SectionType <= MCSectionMachO::LAST_KNOWN_SECTION_TYPE &&
         "Invalid SectionType specified!");

  if (SectionTypeDescriptors[SectionType].AssemblerName) {
    OS << ',';
    OS << SectionTypeDescriptors[SectionType].AssemblerName;
  } else {
    // If we have no name for the attribute, stop here.
    OS << '\n';
    return;
  }

  // If we don't have any attributes, we're done.
  unsigned SectionAttrs = TAA & MCSectionMachO::SECTION_ATTRIBUTES;
  if (SectionAttrs == 0) {
    // If we have a S_SYMBOL_STUBS size specified, print it along with 'none' as
    // the attribute specifier.
    if (Reserved2 != 0)
      OS << ",none," << Reserved2;
    OS << '\n';
    return;
  }

  // Check each attribute to see if we have it.
  char Separator = ',';
  for (unsigned i = 0;
       SectionAttrs != 0 && SectionAttrDescriptors[i].AttrFlag;
       ++i) {
    // Check to see if we have this attribute.
    if ((SectionAttrDescriptors[i].AttrFlag & SectionAttrs) == 0)
      continue;

    // Yep, clear it and print it.
    SectionAttrs &= ~SectionAttrDescriptors[i].AttrFlag;

    OS << Separator;
    if (SectionAttrDescriptors[i].AssemblerName)
      OS << SectionAttrDescriptors[i].AssemblerName;
    else
      OS << "<<" << SectionAttrDescriptors[i].EnumName << ">>";
    Separator = '+';
  }

  assert(SectionAttrs == 0 && "Unknown section attributes!");

  // If we have a S_SYMBOL_STUBS size specified, print it.
  if (Reserved2 != 0)
    OS << ',' << Reserved2;
  OS << '\n';
}

bool MCSectionMachO::UseCodeAlign() const {
  return hasAttribute(MCSectionMachO::S_ATTR_PURE_INSTRUCTIONS);
}

bool MCSectionMachO::isVirtualSection() const {
  return (getType() == MCSectionMachO::S_ZEROFILL ||
          getType() == MCSectionMachO::S_GB_ZEROFILL ||
          getType() == MCSectionMachO::S_THREAD_LOCAL_ZEROFILL);
}

/// StripSpaces - This removes leading and trailing spaces from the StringRef.
static void StripSpaces(StringRef &Str) {
  while (!Str.empty() && isspace(Str[0]))
    Str = Str.substr(1);
  while (!Str.empty() && isspace(Str.back()))
    Str = Str.substr(0, Str.size()-1);
}

/// ParseSectionSpecifier - Parse the section specifier indicated by "Spec".
/// This is a string that can appear after a .section directive in a mach-o
/// flavored .s file.  If successful, this fills in the specified Out
/// parameters and returns an empty string.  When an invalid section
/// specifier is present, this returns a string indicating the problem.
std::string MCSectionMachO::ParseSectionSpecifier(StringRef Spec,        // In.
                                                  StringRef &Segment,    // Out.
                                                  StringRef &Section,    // Out.
                                                  unsigned  &TAA,        // Out.
                                                  bool      &TAAParsed,  // Out.
                                                  unsigned  &StubSize) { // Out.
  TAAParsed = false;
  // Find the first comma.
  std::pair<StringRef, StringRef> Comma = Spec.split(',');

  // If there is no comma, we fail.
  if (Comma.second.empty())
    return "mach-o section specifier requires a segment and section "
           "separated by a comma";

  // Capture segment, remove leading and trailing whitespace.
  Segment = Comma.first;
  StripSpaces(Segment);

  // Verify that the segment is present and not too long.
  if (Segment.empty() || Segment.size() > 16)
    return "mach-o section specifier requires a segment whose length is "
           "between 1 and 16 characters";

  // Split the section name off from any attributes if present.
  Comma = Comma.second.split(',');

  // Capture section, remove leading and trailing whitespace.
  Section = Comma.first;
  StripSpaces(Section);

  // Verify that the section is present and not too long.
  if (Section.empty() || Section.size() > 16)
    return "mach-o section specifier requires a section whose length is "
           "between 1 and 16 characters";

  // If there is no comma after the section, we're done.
  TAA = 0;
  StubSize = 0;
  if (Comma.second.empty())
    return "";

  // Otherwise, we need to parse the section type and attributes.
  Comma = Comma.second.split(',');

  // Get the section type.
  StringRef SectionType = Comma.first;
  StripSpaces(SectionType);

  // Figure out which section type it is.
  unsigned TypeID;
  for (TypeID = 0; TypeID !=MCSectionMachO::LAST_KNOWN_SECTION_TYPE+1; ++TypeID)
    if (SectionTypeDescriptors[TypeID].AssemblerName &&
        SectionType == SectionTypeDescriptors[TypeID].AssemblerName)
      break;

  // If we didn't find the section type, reject it.
  if (TypeID > MCSectionMachO::LAST_KNOWN_SECTION_TYPE)
    return "mach-o section specifier uses an unknown section type";

  // Remember the TypeID.
  TAA = TypeID;
  TAAParsed = true;

  // If we have no comma after the section type, there are no attributes.
  if (Comma.second.empty()) {
    // S_SYMBOL_STUBS always require a symbol stub size specifier.
    if (TAA == MCSectionMachO::S_SYMBOL_STUBS)
      return "mach-o section specifier of type 'symbol_stubs' requires a size "
             "specifier";
    return "";
  }

  // Otherwise, we do have some attributes.  Split off the size specifier if
  // present.
  Comma = Comma.second.split(',');
  StringRef Attrs = Comma.first;

  // The attribute list is a '+' separated list of attributes.
  std::pair<StringRef, StringRef> Plus = Attrs.split('+');

  while (1) {
    StringRef Attr = Plus.first;
    StripSpaces(Attr);

    // Look up the attribute.
    for (unsigned i = 0; ; ++i) {
      if (SectionAttrDescriptors[i].AttrFlag == AttrFlagEnd)
        return "mach-o section specifier has invalid attribute";

      if (SectionAttrDescriptors[i].AssemblerName &&
          Attr == SectionAttrDescriptors[i].AssemblerName) {
        TAA |= SectionAttrDescriptors[i].AttrFlag;
        break;
      }
    }

    if (Plus.second.empty()) break;
    Plus = Plus.second.split('+');
  };

  // Okay, we've parsed the section attributes, see if we have a stub size spec.
  if (Comma.second.empty()) {
    // S_SYMBOL_STUBS always require a symbol stub size specifier.
    if (TAA == MCSectionMachO::S_SYMBOL_STUBS)
      return "mach-o section specifier of type 'symbol_stubs' requires a size "
      "specifier";
    return "";
  }

  // If we have a stub size spec, we must have a sectiontype of S_SYMBOL_STUBS.
  if ((TAA & MCSectionMachO::SECTION_TYPE) != MCSectionMachO::S_SYMBOL_STUBS)
    return "mach-o section specifier cannot have a stub size specified because "
           "it does not have type 'symbol_stubs'";

  // Okay, if we do, it must be a number.
  StringRef StubSizeStr = Comma.second;
  StripSpaces(StubSizeStr);

  // Convert the stub size from a string to an integer.
  if (StubSizeStr.getAsInteger(0, StubSize))
    return "mach-o section specifier has a malformed stub size";

  return "";
}
