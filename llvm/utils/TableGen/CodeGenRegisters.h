//===- CodeGenRegisters.h - Register and RegisterClass Info -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines structures to encapsulate information gleaned from the
// target register and register class definitions.
//
//===----------------------------------------------------------------------===//

#ifndef CODEGEN_REGISTERS_H
#define CODEGEN_REGISTERS_H

#include "llvm/CodeGen/ValueTypes.h"
#include <string>
#include <vector>
#include <cstdlib>

namespace llvm {
  class Record;

  /// CodeGenRegister - Represents a register definition.
  struct CodeGenRegister {
    Record *TheDef;
    const std::string &getName() const;
    unsigned DeclaredSpillSize, DeclaredSpillAlignment;
    CodeGenRegister(Record *R);
  };


  struct CodeGenRegisterClass {
    Record *TheDef;
    std::string Namespace;
    std::vector<Record*> Elements;
    std::vector<MVT::SimpleValueType> VTs;
    unsigned SpillSize;
    unsigned SpillAlignment;
    int CopyCost;
    std::vector<Record*> SubRegClasses;
    std::string MethodProtos, MethodBodies;

    const std::string &getName() const;
    const std::vector<MVT::SimpleValueType> &getValueTypes() const {return VTs;}
    unsigned getNumValueTypes() const { return VTs.size(); }
    
    MVT::SimpleValueType getValueTypeNum(unsigned VTNum) const {
      if (VTNum < VTs.size())
        return VTs[VTNum];
      assert(0 && "VTNum greater than number of ValueTypes in RegClass!");
      abort();
    }

    CodeGenRegisterClass(Record *R);
  };
}

#endif
