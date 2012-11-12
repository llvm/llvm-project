//===- MCSectionMachO.h - MachO Machine Code Sections -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the MCSectionMachO class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCSECTIONMACHO_H
#define LLVM_MC_MCSECTIONMACHO_H

#include "llvm/MC/MCSection.h"
#include "llvm/ADT/StringRef.h"

namespace llvm {

/// MCSectionMachO - This represents a section on a Mach-O system (used by
/// Mac OS X).  On a Mac system, these are also described in
/// /usr/include/mach-o/loader.h.
class MCSectionMachO : public MCSection {
  char SegmentName[16];  // Not necessarily null terminated!
  char SectionName[16];  // Not necessarily null terminated!

  /// TypeAndAttributes - This is the SECTION_TYPE and SECTION_ATTRIBUTES
  /// field of a section, drawn from the enums below.
  unsigned TypeAndAttributes;

  /// Reserved2 - The 'reserved2' field of a section, used to represent the
  /// size of stubs, for example.
  unsigned Reserved2;

  MCSectionMachO(StringRef Segment, StringRef Section,
                 unsigned TAA, unsigned reserved2, SectionKind K);
  friend class MCContext;
public:

  /// These are the section type and attributes fields.  A MachO section can
  /// have only one Type, but can have any of the attributes specified.
  enum {
    // TypeAndAttributes bitmasks.
    SECTION_TYPE       = 0x000000FFU,
    SECTION_ATTRIBUTES = 0xFFFFFF00U,

    // Valid section types.

    /// S_REGULAR - Regular section.
    S_REGULAR                    = 0x00U,
    /// S_ZEROFILL - Zero fill on demand section.
    S_ZEROFILL                   = 0x01U,
    /// S_CSTRING_LITERALS - Section with literal C strings.
    S_CSTRING_LITERALS           = 0x02U,
    /// S_4BYTE_LITERALS - Section with 4 byte literals.
    S_4BYTE_LITERALS             = 0x03U,
    /// S_8BYTE_LITERALS - Section with 8 byte literals.
    S_8BYTE_LITERALS             = 0x04U,
    /// S_LITERAL_POINTERS - Section with pointers to literals.
    S_LITERAL_POINTERS           = 0x05U,
    /// S_NON_LAZY_SYMBOL_POINTERS - Section with non-lazy symbol pointers.
    S_NON_LAZY_SYMBOL_POINTERS   = 0x06U,
    /// S_LAZY_SYMBOL_POINTERS - Section with lazy symbol pointers.
    S_LAZY_SYMBOL_POINTERS       = 0x07U,
    /// S_SYMBOL_STUBS - Section with symbol stubs, byte size of stub in
    /// the Reserved2 field.
    S_SYMBOL_STUBS               = 0x08U,
    /// S_MOD_INIT_FUNC_POINTERS - Section with only function pointers for
    /// initialization.
    S_MOD_INIT_FUNC_POINTERS     = 0x09U,
    /// S_MOD_TERM_FUNC_POINTERS - Section with only function pointers for
    /// termination.
    S_MOD_TERM_FUNC_POINTERS     = 0x0AU,
    /// S_COALESCED - Section contains symbols that are to be coalesced.
    S_COALESCED                  = 0x0BU,
    /// S_GB_ZEROFILL - Zero fill on demand section (that can be larger than 4
    /// gigabytes).
    S_GB_ZEROFILL                = 0x0CU,
    /// S_INTERPOSING - Section with only pairs of function pointers for
    /// interposing.
    S_INTERPOSING                = 0x0DU,
    /// S_16BYTE_LITERALS - Section with only 16 byte literals.
    S_16BYTE_LITERALS            = 0x0EU,
    /// S_DTRACE_DOF - Section contains DTrace Object Format.
    S_DTRACE_DOF                 = 0x0FU,
    /// S_LAZY_DYLIB_SYMBOL_POINTERS - Section with lazy symbol pointers to
    /// lazy loaded dylibs.
    S_LAZY_DYLIB_SYMBOL_POINTERS = 0x10U,
    /// S_THREAD_LOCAL_REGULAR - Section with ....
    S_THREAD_LOCAL_REGULAR = 0x11U,
    /// S_THREAD_LOCAL_ZEROFILL - Thread local zerofill section.
    S_THREAD_LOCAL_ZEROFILL = 0x12U,
    /// S_THREAD_LOCAL_VARIABLES - Section with thread local variable structure
    /// data.
    S_THREAD_LOCAL_VARIABLES = 0x13U,
    /// S_THREAD_LOCAL_VARIABLE_POINTERS - Section with ....
    S_THREAD_LOCAL_VARIABLE_POINTERS = 0x14U,
    /// S_THREAD_LOCAL_INIT_FUNCTION_POINTERS - Section with thread local
    /// variable initialization pointers to functions.
    S_THREAD_LOCAL_INIT_FUNCTION_POINTERS = 0x15U,

    LAST_KNOWN_SECTION_TYPE = S_THREAD_LOCAL_INIT_FUNCTION_POINTERS,


    // Valid section attributes.

    /// S_ATTR_PURE_INSTRUCTIONS - Section contains only true machine
    /// instructions.
    S_ATTR_PURE_INSTRUCTIONS   = 1U << 31,
    /// S_ATTR_NO_TOC - Section contains coalesced symbols that are not to be
    /// in a ranlib table of contents.
    S_ATTR_NO_TOC              = 1U << 30,
    /// S_ATTR_STRIP_STATIC_SYMS - Ok to strip static symbols in this section
    /// in files with the MY_DYLDLINK flag.
    S_ATTR_STRIP_STATIC_SYMS   = 1U << 29,
    /// S_ATTR_NO_DEAD_STRIP - No dead stripping.
    S_ATTR_NO_DEAD_STRIP       = 1U << 28,
    /// S_ATTR_LIVE_SUPPORT - Blocks are live if they reference live blocks.
    S_ATTR_LIVE_SUPPORT        = 1U << 27,
    /// S_ATTR_SELF_MODIFYING_CODE - Used with i386 code stubs written on by
    /// dyld.
    S_ATTR_SELF_MODIFYING_CODE = 1U << 26,
    /// S_ATTR_DEBUG - A debug section.
    S_ATTR_DEBUG               = 1U << 25,
    /// S_ATTR_SOME_INSTRUCTIONS - Section contains some machine instructions.
    S_ATTR_SOME_INSTRUCTIONS   = 1U << 10,
    /// S_ATTR_EXT_RELOC - Section has external relocation entries.
    S_ATTR_EXT_RELOC           = 1U << 9,
    /// S_ATTR_LOC_RELOC - Section has local relocation entries.
    S_ATTR_LOC_RELOC           = 1U << 8
  };

  StringRef getSegmentName() const {
    // SegmentName is not necessarily null terminated!
    if (SegmentName[15])
      return StringRef(SegmentName, 16);
    return StringRef(SegmentName);
  }
  StringRef getSectionName() const {
    // SectionName is not necessarily null terminated!
    if (SectionName[15])
      return StringRef(SectionName, 16);
    return StringRef(SectionName);
  }

  unsigned getTypeAndAttributes() const { return TypeAndAttributes; }
  unsigned getStubSize() const { return Reserved2; }

  unsigned getType() const { return TypeAndAttributes & SECTION_TYPE; }
  bool hasAttribute(unsigned Value) const {
    return (TypeAndAttributes & Value) != 0;
  }

  /// ParseSectionSpecifier - Parse the section specifier indicated by "Spec".
  /// This is a string that can appear after a .section directive in a mach-o
  /// flavored .s file.  If successful, this fills in the specified Out
  /// parameters and returns an empty string.  When an invalid section
  /// specifier is present, this returns a string indicating the problem.
  /// If no TAA was parsed, TAA is not altered, and TAAWasSet becomes false.
  static std::string ParseSectionSpecifier(StringRef Spec,       // In.
                                           StringRef &Segment,   // Out.
                                           StringRef &Section,   // Out.
                                           unsigned  &TAA,       // Out.
                                           bool      &TAAParsed, // Out.
                                           unsigned  &StubSize); // Out.

  virtual void PrintSwitchToSection(const MCAsmInfo &MAI,
                                    raw_ostream &OS) const;
  virtual bool UseCodeAlign() const;
  virtual bool isVirtualSection() const;

  static bool classof(const MCSection *S) {
    return S->getVariant() == SV_MachO;
  }
};

} // end namespace llvm

#endif
