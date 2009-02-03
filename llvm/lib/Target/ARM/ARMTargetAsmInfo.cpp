//===-- ARMTargetAsmInfo.cpp - ARM asm properties ---------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations of the ARMTargetAsmInfo properties.
//
//===----------------------------------------------------------------------===//

#include "ARMTargetAsmInfo.h"
#include "ARMTargetMachine.h"
#include <cstring>
#include <cctype>
using namespace llvm;


const char *const llvm::arm_asm_table[] = {
                                      "{r0}", "r0",
                                      "{r1}", "r1",
                                      "{r2}", "r2",
                                      "{r3}", "r3",
                                      "{r4}", "r4",
                                      "{r5}", "r5",
                                      "{r6}", "r6",
                                      "{r7}", "r7",
                                      "{r8}", "r8",
                                      "{r9}", "r9",
                                      "{r10}", "r10",
                                      "{r11}", "r11",
                                      "{r12}", "r12",
                                      "{r13}", "r13",
                                      "{r14}", "r14",
                                      "{lr}", "lr",
                                      "{sp}", "sp",
                                      "{ip}", "ip",
                                      "{fp}", "fp",
                                      "{sl}", "sl",
                                      "{memory}", "memory",
                                      "{cc}", "cc",
                                      0,0};

ARMDarwinTargetAsmInfo::ARMDarwinTargetAsmInfo(const ARMTargetMachine &TM):
  ARMTargetAsmInfo<DarwinTargetAsmInfo>(TM) {
  Subtarget = &TM.getSubtarget<ARMSubtarget>();

  GlobalPrefix = "_";
  PrivateGlobalPrefix = "L";
  LessPrivateGlobalPrefix = "l";
  StringConstantPrefix = "\1LC";
  BSSSection = 0;                       // no BSS section
  ZeroDirective = "\t.space\t";
  ZeroFillDirective = "\t.zerofill\t";  // Uses .zerofill
  SetDirective = "\t.set\t";
  WeakRefDirective = "\t.weak_reference\t";
  WeakDefDirective = "\t.weak_definition ";
  HiddenDirective = "\t.private_extern\t";
  ProtectedDirective = NULL;
  JumpTableDataSection = ".const";
  CStringSection = "\t.cstring";
  HasDotTypeDotSizeDirective = false;
  HasSingleParameterDotFile = false;
  NeedsIndirectEncoding = true;
  if (TM.getRelocationModel() == Reloc::Static) {
    StaticCtorsSection = ".constructor";
    StaticDtorsSection = ".destructor";
  } else {
    StaticCtorsSection = ".mod_init_func";
    StaticDtorsSection = ".mod_term_func";
  }

  // In non-PIC modes, emit a special label before jump tables so that the
  // linker can perform more accurate dead code stripping.
  if (TM.getRelocationModel() != Reloc::PIC_) {
    // Emit a local label that is preserved until the linker runs.
    JumpTableSpecialLabelPrefix = "l";
  }

  NeedsSet = true;
  DwarfAbbrevSection = ".section __DWARF,__debug_abbrev,regular,debug";
  DwarfInfoSection = ".section __DWARF,__debug_info,regular,debug";
  DwarfLineSection = ".section __DWARF,__debug_line,regular,debug";
  DwarfFrameSection = ".section __DWARF,__debug_frame,regular,debug";
  DwarfPubNamesSection = ".section __DWARF,__debug_pubnames,regular,debug";
  DwarfPubTypesSection = ".section __DWARF,__debug_pubtypes,regular,debug";
  DwarfStrSection = ".section __DWARF,__debug_str,regular,debug";
  DwarfLocSection = ".section __DWARF,__debug_loc,regular,debug";
  DwarfARangesSection = ".section __DWARF,__debug_aranges,regular,debug";
  DwarfRangesSection = ".section __DWARF,__debug_ranges,regular,debug";
  DwarfMacInfoSection = ".section __DWARF,__debug_macinfo,regular,debug";
}

ARMELFTargetAsmInfo::ARMELFTargetAsmInfo(const ARMTargetMachine &TM):
  ARMTargetAsmInfo<ELFTargetAsmInfo>(TM) {
  Subtarget = &TM.getSubtarget<ARMSubtarget>();

  NeedsSet = false;
  HasLEB128 = true;
  AbsoluteDebugSectionOffsets = true;
  CStringSection = ".rodata.str";
  PrivateGlobalPrefix = ".L";
  WeakRefDirective = "\t.weak\t";
  SetDirective = "\t.set\t";
  DwarfRequiresFrameSection = false;
  DwarfAbbrevSection =  "\t.section\t.debug_abbrev,\"\",%progbits";
  DwarfInfoSection =    "\t.section\t.debug_info,\"\",%progbits";
  DwarfLineSection =    "\t.section\t.debug_line,\"\",%progbits";
  DwarfFrameSection =   "\t.section\t.debug_frame,\"\",%progbits";
  DwarfPubNamesSection ="\t.section\t.debug_pubnames,\"\",%progbits";
  DwarfPubTypesSection ="\t.section\t.debug_pubtypes,\"\",%progbits";
  DwarfStrSection =     "\t.section\t.debug_str,\"\",%progbits";
  DwarfLocSection =     "\t.section\t.debug_loc,\"\",%progbits";
  DwarfARangesSection = "\t.section\t.debug_aranges,\"\",%progbits";
  DwarfRangesSection =  "\t.section\t.debug_ranges,\"\",%progbits";
  DwarfMacInfoSection = "\t.section\t.debug_macinfo,\"\",%progbits";

  if (Subtarget->isAAPCS_ABI()) {
    StaticCtorsSection = "\t.section .init_array,\"aw\",%init_array";
    StaticDtorsSection = "\t.section .fini_array,\"aw\",%fini_array";
  } else {
    StaticCtorsSection = "\t.section .ctors,\"aw\",%progbits";
    StaticDtorsSection = "\t.section .dtors,\"aw\",%progbits";
  }
}

/// Count the number of comma-separated arguments.
/// Do not try to detect errors.
template <class BaseTAI>
unsigned ARMTargetAsmInfo<BaseTAI>::countArguments(const char* p) const {
  unsigned count = 0;
  while (*p && isspace(*p) && *p != '\n')
    p++;
  count++;
  while (*p && *p!='\n' &&
         strncmp(p, BaseTAI::CommentString,
                 strlen(BaseTAI::CommentString))!=0) {
    if (*p==',')
      count++;
    p++;
  }
  return count;
}

/// Count the length of a string enclosed in quote characters.
/// Do not try to detect errors.
template <class BaseTAI>
unsigned ARMTargetAsmInfo<BaseTAI>::countString(const char* p) const {
  unsigned count = 0;
  while (*p && isspace(*p) && *p!='\n')
    p++;
  if (!*p || *p != '\"')
    return count;
  while (*++p && *p != '\"')
    count++;
  return count;
}

/// ARM-specific version of TargetAsmInfo::getInlineAsmLength.
template <class BaseTAI>
unsigned ARMTargetAsmInfo<BaseTAI>::getInlineAsmLength(const char *s) const {
  // Make a lowercase-folded version of s for counting purposes.
  char *q, *s_copy = (char *)malloc(strlen(s) + 1);
  strcpy(s_copy, s);
  for (q=s_copy; *q; q++)
    *q = tolower(*q);
  const char *Str = s_copy;

  // Count the number of bytes in the asm.
  bool atInsnStart = true;
  bool inTextSection = true;
  unsigned Length = 0;
  for (; *Str; ++Str) {
    if (atInsnStart) {
      // Skip whitespace
      while (*Str && isspace(*Str) && *Str != '\n')
        Str++;
      // Skip label
      for (const char* p = Str; *p && !isspace(*p); p++)
        if (*p == ':') {
          Str = p+1;
          while (*Str && isspace(*Str) && *Str != '\n')
            Str++;
          break;
        }
      // Ignore everything from comment char(s) to EOL
      if (strncmp(Str, BaseTAI::CommentString, strlen(BaseTAI::CommentString))==-0)
        atInsnStart = false;
      // FIXME do something like the following for non-Darwin
      else if (*Str == '.' && Subtarget->isTargetDarwin()) {
        // Directive.
        atInsnStart = false;

        // Some change the section, but don't generate code.
        if (strncmp(Str, ".literal4", strlen(".literal4"))==0 ||
            strncmp(Str, ".literal8", strlen(".literal8"))==0 ||
            strncmp(Str, ".const", strlen(".const"))==0 ||
            strncmp(Str, ".constructor", strlen(".constructor"))==0 ||
            strncmp(Str, ".cstring", strlen(".cstring"))==0 ||
            strncmp(Str, ".data", strlen(".data"))==0 ||
            strncmp(Str, ".destructor", strlen(".destructor"))==0 ||
            strncmp(Str, ".fvmlib_init0", strlen(".fvmlib_init0"))==0 ||
            strncmp(Str, ".fvmlib_init1", strlen(".fvmlib_init1"))==0 ||
            strncmp(Str, ".mod_init_func", strlen(".mod_init_func"))==0 ||
            strncmp(Str, ".mod_term_func", strlen(".mod_term_func"))==0 ||
            strncmp(Str, ".picsymbol_stub", strlen(".picsymbol_stub"))==0 ||
            strncmp(Str, ".symbol_stub", strlen(".symbol_stub"))==0 ||
            strncmp(Str, ".static_data", strlen(".static_data"))==0 ||
            strncmp(Str, ".section", strlen(".section"))==0 ||
            strncmp(Str, ".lazy_symbol_pointer", strlen(".lazy_symbol_pointer"))==0 ||
            strncmp(Str, ".non_lazy_symbol_pointer", strlen(".non_lazy_symbol_pointer"))==0 ||
            strncmp(Str, ".dyld", strlen(".dyld"))==0 ||
            strncmp(Str, ".const_data", strlen(".const_data"))==0 ||
            strncmp(Str, ".objc", strlen(".objc"))==0 ||       //// many directives
            strncmp(Str, ".static_const", strlen(".static_const"))==0)
          inTextSection=false;
        else if (strncmp(Str, ".text", strlen(".text"))==0)
          inTextSection = true;
        // Some can't really be handled without implementing significant pieces
        // of an assembler.  Others require dynamic adjustment of block sizes in
        // AdjustBBOffsetsAfter; it's a big compile-time speed hit to check every
        // instruction in there, and none of these are currently used in the kernel.
        else if (strncmp(Str, ".macro", strlen(".macro"))==0 ||
                 strncmp(Str, ".if", strlen(".if"))==0 ||
                 strncmp(Str, ".align", strlen(".align"))==0 ||
                 strncmp(Str, ".fill", strlen(".fill"))==0 ||
                 strncmp(Str, ".space", strlen(".space"))==0 ||
                 strncmp(Str, ".zerofill", strlen(".zerofill"))==0 ||
                 strncmp(Str, ".p2align", strlen(".p2align"))==0 ||
                 strncmp(Str, ".p2alignw", strlen(".p2alignw"))==0 ||
                 strncmp(Str, ".p2alignl", strlen(".p2alignl"))==0 ||
                 strncmp(Str, ".align32", strlen(".p2align32"))==0 ||
                 strncmp(Str, ".include", strlen(".include"))==0)
          cerr << "Directive " << Str << " in asm may lead to invalid offsets for" <<
                   " constant pools (the assembler will tell you if this happens).\n";
        // Some generate code, but this is only interesting in the text section.
        else if (inTextSection) {
          if (strncmp(Str, ".long", strlen(".long"))==0)
            Length += 4*countArguments(Str+strlen(".long"));
          else if (strncmp(Str, ".short", strlen(".short"))==0)
            Length += 2*countArguments(Str+strlen(".short"));
          else if (strncmp(Str, ".byte", strlen(".byte"))==0)
            Length += 1*countArguments(Str+strlen(".byte"));
          else if (strncmp(Str, ".single", strlen(".single"))==0)
            Length += 4*countArguments(Str+strlen(".single"));
          else if (strncmp(Str, ".double", strlen(".double"))==0)
            Length += 8*countArguments(Str+strlen(".double"));
          else if (strncmp(Str, ".quad", strlen(".quad"))==0)
            Length += 16*countArguments(Str+strlen(".quad"));
          else if (strncmp(Str, ".ascii", strlen(".ascii"))==0)
            Length += countString(Str+strlen(".ascii"));
          else if (strncmp(Str, ".asciz", strlen(".asciz"))==0)
            Length += countString(Str+strlen(".asciz"))+1;
        }
      } else if (inTextSection) {
        // An instruction
        atInsnStart = false;
        if (Subtarget->isThumb()) {
          // BL and BLX <non-reg> are 4 bytes, all others 2.
          if (strncmp(Str, "blx", strlen("blx"))==0) {
            const char* p = Str+3;
            while (*p && isspace(*p))
              p++;
            if (*p == 'r' || *p=='R')
              Length += 2;    // BLX reg
            else
              Length += 4;    // BLX non-reg
          } else if (strncmp(Str, "bl", strlen("bl"))==0)
            Length += 4;    // BL
          else
            Length += 2;    // Thumb anything else
        }
        else
          Length += 4;    // ARM
      }
    }
    if (*Str == '\n' || *Str == BaseTAI::SeparatorChar)
      atInsnStart = true;
  }
  free(s_copy);
  return Length;
}

// Instantiate default implementation.
TEMPLATE_INSTANTIATION(class ARMTargetAsmInfo<TargetAsmInfo>);
