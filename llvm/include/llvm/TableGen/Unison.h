//===- llvm/TableGen/Unison.h - Unison tool ---------------------*- C++ -*-===//
//
//  Main authors:
//    Jan Tomljanovic <jan.tomljanovic@sics.se>
//    Roberto Castaneda Lozano <roberto.castaneda@ri.se>
//
//  This file is part of Unison, see http://unison-code.github.io
//
//  Copyright (c) 2016, RISE SICS AB
//  All rights reserved.
//
//  Redistribution and use in source and binary forms, with or without
//  modification, are permitted provided that the following conditions are met:
//  1. Redistributions of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//  2. Redistributions in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//  3. Neither the name of the copyright holder nor the names of its
//     contributors may be used to endorse or promote products derived from this
//     software without specific prior written permission.
//
//  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
//  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
//  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
//  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
//  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
//  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
//  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
//  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
//  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
//  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
//  POSSIBILITY OF SUCH DAMAGE.
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Extraction of the following information about each instruction for Unison:
///   - id (opcode)
///   - type (linear, call, or branch)
///   - operands (including use/def information and reg. class, if applicable)
///   - size
///   - side effects (including memory reads and writes)
///   - itinerary
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TABLEGEN_UNISON_H
#define LLVM_TABLEGEN_UNISON_H

#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Record.h"
#include <string>
#include <utility>
#include <vector>

// An Operand can be a Register, Label or a Bound (any other Operand that is not
// interpreted by Unison, such as immediates). If it is a register, UseDef and
// RegType are defined.
namespace unison {

struct Operand {
  enum { Register, Label, Bound } Type;
  std::string Name;
  std::string UseDef;
  std::string RegType;
};

} // end namespace unison

typedef std::pair<std::string, std::string> StringPair;
typedef std::vector<StringPair> StringPairVector;
typedef std::vector<std::string> StringVector;
typedef std::vector<unison::Operand> OperandVector;

/// Instruction with methods to be printed in .yaml format.
class Instruction {
private:
  std::string Id;
  std::string Type;
  OperandVector Operands;
  StringVector Uses;
  StringVector Defs;
  int Size;
  bool AffectsMem;
  bool AffectedMem;
  StringVector AffectsReg;
  StringVector AffectedReg;
  std::string Itinerary;

  void printAffs(llvm::raw_ostream &OS, std::string Name, bool Memory,
                 StringVector Regs);
  void printUseDefs(llvm::raw_ostream &OS, StringVector UseDefs,
                    std::string Name);
  void printAttribute(std::string Name, std::string Value,
                      llvm::raw_ostream &OS);
  void printField(std::string Name, std::string Value, llvm::raw_ostream &OS);

public:
  Instruction(std::string Id, std::string Type, OperandVector Operands,
              StringVector Uses, StringVector Defs, int Size, bool AffectsMem,
              bool AffectedMem, StringVector AffectsReg,
              StringVector AffectedReg, std::string Itinerary);
  void printId(llvm::raw_ostream &OS);
  void printType(llvm::raw_ostream &OS);
  void printOperands(llvm::raw_ostream &OS);
  void printUses(llvm::raw_ostream &OS);
  void printDefs(llvm::raw_ostream &OS);
  void printSize(llvm::raw_ostream &OS);
  void printAffects(llvm::raw_ostream &OS);
  void printAffected(llvm::raw_ostream &OS);
  void printItinerary(llvm::raw_ostream &OS);
  void printAll(llvm::raw_ostream &OS);
};

namespace llvm {

/// \brief outputs information for Unison.
///
/// Prints extracted information for the Unison compiler as a valid
/// .yaml file.
/// \param OS output stream to which it prints the .yaml file.
/// \param Records structure that holds all the information about the
/// data which TableGen tool has.
void EmitUnisonFile(const RecordKeeper &Records, raw_ostream &OS);

StringVector flat(const Record *Rec);
void printYaml(std::vector<Instruction> Instructions, raw_ostream &OS);
std::string getRecordItinerary(Record *Rec);
StringVector getRegisterList(std::string Field, Record *Rec);
bool getRecordBool(Record *Rec, std::string Field, bool Def);
int getRecordSize(Record *Rec);
StringPairVector *parseOperands(std::string Field, Record *Rec);
StringVector getNames(StringPairVector *List);
void executeConstraints(StringPairVector *Outs, std::string Cons);
OperandVector getOperands(StringPairVector *Outs, StringPairVector *ins,
                          const RecordKeeper &Records);
void getOperandsFromVector(StringPairVector *Vec, StringPairVector *Help,
                           OperandVector *Operands, bool Defs,
                           const RecordKeeper &Records);
bool isRegister(const Record *Rec);
bool isLabel(const Record *Rec);
std::string getRecordType(Record *Rec);
std::string getRecordId(Record *Rec);
bool fieldExists(Record *Rec, std::string Field);
bool allNeededFieldsExist(Record *Rec);
StringVector split(std::string Str, char Del);
std::string trim(std::string Str);
std::string escape(std::string Name);

} // end namespace llvm

#endif
