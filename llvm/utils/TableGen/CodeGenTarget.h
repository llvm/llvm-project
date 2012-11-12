//===- CodeGenTarget.h - Target Class Wrapper -------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines wrappers for the Target class and related global
// functionality.  This makes it easier to access the data and provides a single
// place that needs to check it for validity.  All of these classes abort
// on error conditions.
//
//===----------------------------------------------------------------------===//

#ifndef CODEGEN_TARGET_H
#define CODEGEN_TARGET_H

#include "CodeGenRegisters.h"
#include "CodeGenInstruction.h"
#include "llvm/TableGen/Record.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>

namespace llvm {

struct CodeGenRegister;
class CodeGenSchedModels;
class CodeGenTarget;

// SelectionDAG node properties.
//  SDNPMemOperand: indicates that a node touches memory and therefore must
//                  have an associated memory operand that describes the access.
enum SDNP {
  SDNPCommutative,
  SDNPAssociative,
  SDNPHasChain,
  SDNPOutGlue,
  SDNPInGlue,
  SDNPOptInGlue,
  SDNPMayLoad,
  SDNPMayStore,
  SDNPSideEffect,
  SDNPMemOperand,
  SDNPVariadic,
  SDNPWantRoot,
  SDNPWantParent
};

/// getValueType - Return the MVT::SimpleValueType that the specified TableGen
/// record corresponds to.
MVT::SimpleValueType getValueType(Record *Rec);

std::string getName(MVT::SimpleValueType T);
std::string getEnumName(MVT::SimpleValueType T);

/// getQualifiedName - Return the name of the specified record, with a
/// namespace qualifier if the record contains one.
std::string getQualifiedName(const Record *R);

/// CodeGenTarget - This class corresponds to the Target class in the .td files.
///
class CodeGenTarget {
  RecordKeeper &Records;
  Record *TargetRec;

  mutable DenseMap<const Record*, CodeGenInstruction*> Instructions;
  mutable CodeGenRegBank *RegBank;
  mutable std::vector<Record*> RegAltNameIndices;
  mutable std::vector<MVT::SimpleValueType> LegalValueTypes;
  void ReadRegAltNameIndices() const;
  void ReadInstructions() const;
  void ReadLegalValueTypes() const;

  mutable CodeGenSchedModels *SchedModels;

  mutable std::vector<const CodeGenInstruction*> InstrsByEnum;
public:
  CodeGenTarget(RecordKeeper &Records);
  ~CodeGenTarget();

  Record *getTargetRecord() const { return TargetRec; }
  const std::string &getName() const;

  /// getInstNamespace - Return the target-specific instruction namespace.
  ///
  std::string getInstNamespace() const;

  /// getInstructionSet - Return the InstructionSet object.
  ///
  Record *getInstructionSet() const;

  /// getAsmParser - Return the AssemblyParser definition for this target.
  ///
  Record *getAsmParser() const;

  /// getAsmParserVariant - Return the AssmblyParserVariant definition for
  /// this target.
  ///
  Record *getAsmParserVariant(unsigned i) const;

  /// getAsmParserVariantCount - Return the AssmblyParserVariant definition
  /// available for this target.
  ///
  unsigned getAsmParserVariantCount() const;

  /// getAsmWriter - Return the AssemblyWriter definition for this target.
  ///
  Record *getAsmWriter() const;

  /// getRegBank - Return the register bank description.
  CodeGenRegBank &getRegBank() const;

  /// getRegisterByName - If there is a register with the specific AsmName,
  /// return it.
  const CodeGenRegister *getRegisterByName(StringRef Name) const;

  const std::vector<Record*> &getRegAltNameIndices() const {
    if (RegAltNameIndices.empty()) ReadRegAltNameIndices();
    return RegAltNameIndices;
  }

  const CodeGenRegisterClass &getRegisterClass(Record *R) const {
    return *getRegBank().getRegClass(R);
  }

  /// getRegisterVTs - Find the union of all possible SimpleValueTypes for the
  /// specified physical register.
  std::vector<MVT::SimpleValueType> getRegisterVTs(Record *R) const;

  const std::vector<MVT::SimpleValueType> &getLegalValueTypes() const {
    if (LegalValueTypes.empty()) ReadLegalValueTypes();
    return LegalValueTypes;
  }

  /// isLegalValueType - Return true if the specified value type is natively
  /// supported by the target (i.e. there are registers that directly hold it).
  bool isLegalValueType(MVT::SimpleValueType VT) const {
    const std::vector<MVT::SimpleValueType> &LegalVTs = getLegalValueTypes();
    for (unsigned i = 0, e = LegalVTs.size(); i != e; ++i)
      if (LegalVTs[i] == VT) return true;
    return false;
  }

  CodeGenSchedModels &getSchedModels() const;

private:
  DenseMap<const Record*, CodeGenInstruction*> &getInstructions() const {
    if (Instructions.empty()) ReadInstructions();
    return Instructions;
  }
public:

  CodeGenInstruction &getInstruction(const Record *InstRec) const {
    if (Instructions.empty()) ReadInstructions();
    DenseMap<const Record*, CodeGenInstruction*>::iterator I =
      Instructions.find(InstRec);
    assert(I != Instructions.end() && "Not an instruction");
    return *I->second;
  }

  /// getInstructionsByEnumValue - Return all of the instructions defined by the
  /// target, ordered by their enum value.
  const std::vector<const CodeGenInstruction*> &
  getInstructionsByEnumValue() const {
    if (InstrsByEnum.empty()) ComputeInstrsByEnum();
    return InstrsByEnum;
  }

  typedef std::vector<const CodeGenInstruction*>::const_iterator inst_iterator;
  inst_iterator inst_begin() const{return getInstructionsByEnumValue().begin();}
  inst_iterator inst_end() const { return getInstructionsByEnumValue().end(); }


  /// isLittleEndianEncoding - are instruction bit patterns defined as  [0..n]?
  ///
  bool isLittleEndianEncoding() const;

  /// guessInstructionProperties - should we just guess unset instruction
  /// properties?
  bool guessInstructionProperties() const;

private:
  void ComputeInstrsByEnum() const;
};

/// ComplexPattern - ComplexPattern info, corresponding to the ComplexPattern
/// tablegen class in TargetSelectionDAG.td
class ComplexPattern {
  MVT::SimpleValueType Ty;
  unsigned NumOperands;
  std::string SelectFunc;
  std::vector<Record*> RootNodes;
  unsigned Properties; // Node properties
public:
  ComplexPattern() : NumOperands(0) {}
  ComplexPattern(Record *R);

  MVT::SimpleValueType getValueType() const { return Ty; }
  unsigned getNumOperands() const { return NumOperands; }
  const std::string &getSelectFunc() const { return SelectFunc; }
  const std::vector<Record*> &getRootNodes() const {
    return RootNodes;
  }
  bool hasProperty(enum SDNP Prop) const { return Properties & (1 << Prop); }
};

} // End llvm namespace

#endif
