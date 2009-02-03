//=== ELFRelocation.h - ELF Relocation Info ---------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the ELFRelocation class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_ELF_RELOCATION_H
#define LLVM_CODEGEN_ELF_RELOCATION_H

#include "llvm/Support/DataTypes.h"

namespace llvm {

  /// ELFRelocation - This class contains all the information necessary to
  /// to generate any 32-bit or 64-bit ELF relocation entry.
  class ELFRelocation {
    uint64_t r_offset;    // offset in the section of the object this applies to
    uint32_t r_symidx;    // symbol table index of the symbol to use
    uint32_t r_type;      // machine specific relocation type
    int64_t  r_add;       // explicit relocation addend
    bool     r_rela;      // if true then the addend is part of the entry
                          // otherwise the addend is at the location specified
                          // by r_offset
  public:      
  
    uint64_t getInfo(bool is64Bit = false) const {
      if (is64Bit)
        return ((uint64_t)r_symidx << 32) + ((uint64_t)r_type & 0xFFFFFFFFL);
      else
        return (r_symidx << 8)  + (r_type & 0xFFL);
    }
  
    uint64_t getOffset() const { return r_offset; }
    uint64_t getAddress() const { return r_add; }
  
    ELFRelocation(uint64_t off, uint32_t sym, uint32_t type, 
                  bool rela = true, int64_t addend = 0) : 
      r_offset(off), r_symidx(sym), r_type(type),
      r_add(addend), r_rela(rela) {}
  };

} // end llvm namespace

#endif // LLVM_CODEGEN_ELF_RELOCATION_H
