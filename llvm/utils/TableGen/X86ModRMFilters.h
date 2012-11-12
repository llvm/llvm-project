//===- X86ModRMFilters.h - Disassembler ModR/M filterss ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is part of the X86 Disassembler Emitter.
// It contains ModR/M filters that determine which values of the ModR/M byte
//  are valid for a partiuclar instruction.
// Documentation for the disassembler emitter in general can be found in
//  X86DisasemblerEmitter.h.
//
//===----------------------------------------------------------------------===//

#ifndef X86MODRMFILTERS_H
#define X86MODRMFILTERS_H

#include "llvm/Support/DataTypes.h"

namespace llvm {

namespace X86Disassembler {

/// ModRMFilter - Abstract base class for clases that recognize patterns in
///   ModR/M bytes.
class ModRMFilter {
  virtual void anchor();
public:
  /// Destructor    - Override as necessary.
  virtual ~ModRMFilter() { }

  /// isDumb        - Indicates whether this filter returns the same value for
  ///                 any value of the ModR/M byte.
  ///
  /// @result       - True if the filter returns the same value for any ModR/M
  ///                 byte; false if not.
  virtual bool isDumb() const { return false; }
  
  /// accepts       - Indicates whether the filter accepts a particular ModR/M
  ///                 byte value.
  ///
  /// @result       - True if the filter accepts the ModR/M byte; false if not.
  virtual bool accepts(uint8_t modRM) const = 0;
};

/// DumbFilter - Accepts any ModR/M byte.  Used for instructions that do not
///   require a ModR/M byte or instructions where the entire ModR/M byte is used
///   for operands.
class DumbFilter : public ModRMFilter {
  virtual void anchor();
public:
  bool isDumb() const {
    return true;
  }
  
  bool accepts(uint8_t modRM) const {
    return true;
  }
};

/// ModFilter - Filters based on the mod bits [bits 7-6] of the ModR/M byte.
///   Some instructions are classified based on whether they are 11 or anything
///   else.  This filter performs that classification.
class ModFilter : public ModRMFilter {
  virtual void anchor();
  bool R;
public:
  /// Constructor
  ///
  /// \param r        True if the mod bits of the ModR/M byte must be 11; false
  ///                 otherwise.  The name r derives from the fact that the mod
  ///                 bits indicate whether the R/M bits [bits 2-0] signify a
  ///                 register or a memory operand.
  ModFilter(bool r) :
    ModRMFilter(),
    R(r) {
  }
    
  bool accepts(uint8_t modRM) const {
    if (R == ((modRM & 0xc0) == 0xc0))
      return true;
    else
      return false;
  }
};

/// EscapeFilter - Filters escape opcodes, which are classified in two ways.  If
///   the ModR/M byte is between 0xc0 and 0xff, then there is one slot for each
///   possible value.  Otherwise, there is one instruction for each value of the
///   nnn field [bits 5-3], known elsewhere as the reg field.
class EscapeFilter : public ModRMFilter {
  virtual void anchor();
  bool C0_FF;
  uint8_t NNN_or_ModRM;
public:
  /// Constructor
  ///
  /// \param c0_ff True if the ModR/M byte must fall between 0xc0 and 0xff;
  ///              false otherwise.
  ///
  /// \param nnn_or_modRM If c0_ff is true, the required value of the entire
  ///                     ModR/M byte.  If c0_ff is false, the required value
  ///                     of the nnn field.
  EscapeFilter(bool c0_ff, uint8_t nnn_or_modRM) :
    ModRMFilter(),
    C0_FF(c0_ff),
    NNN_or_ModRM(nnn_or_modRM) {
  }
    
  bool accepts(uint8_t modRM) const {
    if ((C0_FF && modRM >= 0xc0 && (modRM == NNN_or_ModRM)) ||
        (!C0_FF && modRM < 0xc0  && ((modRM & 0x38) >> 3) == NNN_or_ModRM))
      return true;
    else
      return false;
  }
};

/// AddRegEscapeFilter - Some escape opcodes have one of the register operands
///   added to the ModR/M byte, meaning that a range of eight ModR/M values
///   maps to a single instruction.  Such instructions require the ModR/M byte
///   to fall between 0xc0 and 0xff.
class AddRegEscapeFilter : public ModRMFilter {
  virtual void anchor();
  uint8_t ModRM;
public:
  /// Constructor
  ///
  /// \param modRM The value of the ModR/M byte when the register operand
  ///              refers to the first register in the register set.
  AddRegEscapeFilter(uint8_t modRM) : ModRM(modRM) {
  }
  
  bool accepts(uint8_t modRM) const {
    if (modRM >= ModRM && modRM < ModRM + 8)
      return true;
    else
      return false;
  }
};

/// ExtendedFilter - Extended opcodes are classified based on the value of the
///   mod field [bits 7-6] and the value of the nnn field [bits 5-3]. 
class ExtendedFilter : public ModRMFilter {
  virtual void anchor();
  bool R;
  uint8_t NNN;
public:
  /// Constructor
  ///
  /// \param r   True if the mod field must be set to 11; false otherwise.
  ///            The name is explained at ModFilter.
  /// \param nnn The required value of the nnn field.
  ExtendedFilter(bool r, uint8_t nnn) : 
    ModRMFilter(),
    R(r),
    NNN(nnn) {
  }
    
  bool accepts(uint8_t modRM) const {
    if (((R  && ((modRM & 0xc0) == 0xc0)) ||
        (!R && ((modRM & 0xc0) != 0xc0))) &&
        (((modRM & 0x38) >> 3) == NNN))
      return true;
    else
      return false;
  }
};

/// ExactFilter - The occasional extended opcode (such as VMCALL or MONITOR)
///   requires the ModR/M byte to have a specific value.
class ExactFilter : public ModRMFilter {
  virtual void anchor();
  uint8_t ModRM;
public:
  /// Constructor
  ///
  /// \param modRM The required value of the full ModR/M byte.
  ExactFilter(uint8_t modRM) :
    ModRMFilter(),
    ModRM(modRM) {
  }
    
  bool accepts(uint8_t modRM) const {
    if (ModRM == modRM)
      return true;
    else
      return false;
  }
};

} // namespace X86Disassembler

} // namespace llvm

#endif
