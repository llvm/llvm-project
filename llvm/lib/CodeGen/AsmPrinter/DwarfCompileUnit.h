//===-- llvm/CodeGen/DwarfCompileUnit.h - Dwarf Compile Unit ---*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains support for writing dwarf compile unit.
//
//===----------------------------------------------------------------------===//

#ifndef CODEGEN_ASMPRINTER_DWARFCOMPILEUNIT_H
#define CODEGEN_ASMPRINTER_DWARFCOMPILEUNIT_H

#include "DIE.h"
#include "llvm/DebugInfo.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/OwningPtr.h"

namespace llvm {

class DwarfDebug;
class MachineLocation;
class MachineOperand;
class ConstantInt;
class DbgVariable;

//===----------------------------------------------------------------------===//
/// CompileUnit - This dwarf writer support class manages information associated
/// with a source file.
class CompileUnit {
  /// ID - File identifier for source.
  ///
  unsigned ID;

  /// Language - The DW_AT_language of the compile unit
  ///
  unsigned Language;

  /// Die - Compile unit debug information entry.
  ///
  const OwningPtr<DIE> CUDie;

  /// Asm - Target of Dwarf emission.
  AsmPrinter *Asm;

  DwarfDebug *DD;

  /// IndexTyDie - An anonymous type for index type.  Owned by CUDie.
  DIE *IndexTyDie;

  /// MDNodeToDieMap - Tracks the mapping of unit level debug informaton
  /// variables to debug information entries.
  DenseMap<const MDNode *, DIE *> MDNodeToDieMap;

  /// MDNodeToDIEEntryMap - Tracks the mapping of unit level debug informaton
  /// descriptors to debug information entries using a DIEEntry proxy.
  DenseMap<const MDNode *, DIEEntry *> MDNodeToDIEEntryMap;

  /// GlobalTypes - A map of globally visible types for this unit.
  ///
  StringMap<DIE*> GlobalTypes;

  /// AccelNames - A map of names for the name accelerator table.
  ///
  StringMap<std::vector<DIE*> > AccelNames;
  StringMap<std::vector<DIE*> > AccelObjC;
  StringMap<std::vector<DIE*> > AccelNamespace;
  StringMap<std::vector<std::pair<DIE*, unsigned> > > AccelTypes;

  /// DIEBlocks - A list of all the DIEBlocks in use.
  std::vector<DIEBlock *> DIEBlocks;

  /// ContainingTypeMap - This map is used to keep track of subprogram DIEs that
  /// need DW_AT_containing_type attribute. This attribute points to a DIE that
  /// corresponds to the MDNode mapped with the subprogram DIE.
  DenseMap<DIE *, const MDNode *> ContainingTypeMap;

public:
  CompileUnit(unsigned I, unsigned L, DIE *D, AsmPrinter *A, DwarfDebug *DW);
  ~CompileUnit();

  // Accessors.
  unsigned getID()                  const { return ID; }
  unsigned getLanguage()            const { return Language; }
  DIE* getCUDie()                   const { return CUDie.get(); }
  const StringMap<DIE*> &getGlobalTypes() const { return GlobalTypes; }

  const StringMap<std::vector<DIE*> > &getAccelNames() const {
    return AccelNames;
  }
  const StringMap<std::vector<DIE*> > &getAccelObjC() const {
    return AccelObjC;
  }
  const StringMap<std::vector<DIE*> > &getAccelNamespace() const {
    return AccelNamespace;
  }
  const StringMap<std::vector<std::pair<DIE*, unsigned > > >
  &getAccelTypes() const {
    return AccelTypes;
  }
  
  /// hasContent - Return true if this compile unit has something to write out.
  ///
  bool hasContent() const { return !CUDie->getChildren().empty(); }

  /// addGlobalType - Add a new global type to the compile unit.
  ///
  void addGlobalType(DIType Ty);


  /// addAccelName - Add a new name to the name accelerator table.
  void addAccelName(StringRef Name, DIE *Die) {
    std::vector<DIE*> &DIEs = AccelNames[Name];
    DIEs.push_back(Die);
  }
  void addAccelObjC(StringRef Name, DIE *Die) {
    std::vector<DIE*> &DIEs = AccelObjC[Name];
    DIEs.push_back(Die);
  }
  void addAccelNamespace(StringRef Name, DIE *Die) {
    std::vector<DIE*> &DIEs = AccelNamespace[Name];
    DIEs.push_back(Die);
  }
  void addAccelType(StringRef Name, std::pair<DIE *, unsigned> Die) {
    std::vector<std::pair<DIE*, unsigned > > &DIEs = AccelTypes[Name];
    DIEs.push_back(Die);
  }
  
  /// getDIE - Returns the debug information entry map slot for the
  /// specified debug variable.
  DIE *getDIE(const MDNode *N) { return MDNodeToDieMap.lookup(N); }

  DIEBlock *getDIEBlock() { 
    return new (DIEValueAllocator) DIEBlock();
  }

  /// insertDIE - Insert DIE into the map.
  void insertDIE(const MDNode *N, DIE *D) {
    MDNodeToDieMap.insert(std::make_pair(N, D));
  }

  /// getDIEEntry - Returns the debug information entry for the specified
  /// debug variable.
  DIEEntry *getDIEEntry(const MDNode *N) {
    DenseMap<const MDNode *, DIEEntry *>::iterator I =
      MDNodeToDIEEntryMap.find(N);
    if (I == MDNodeToDIEEntryMap.end())
      return NULL;
    return I->second;
  }

  /// insertDIEEntry - Insert debug information entry into the map.
  void insertDIEEntry(const MDNode *N, DIEEntry *E) {
    MDNodeToDIEEntryMap.insert(std::make_pair(N, E));
  }

  /// addDie - Adds or interns the DIE to the compile unit.
  ///
  void addDie(DIE *Buffer) {
    this->CUDie->addChild(Buffer);
  }

  // getIndexTyDie - Get an anonymous type for index type.
  DIE *getIndexTyDie() {
    return IndexTyDie;
  }

  // setIndexTyDie - Set D as anonymous type for index which can be reused
  // later.
  void setIndexTyDie(DIE *D) {
    IndexTyDie = D;
  }
public:

  /// addFlag - Add a flag that is true to the DIE.
  void addFlag(DIE *Die, unsigned Attribute);
  
  /// addUInt - Add an unsigned integer attribute data and value.
  ///
  void addUInt(DIE *Die, unsigned Attribute, unsigned Form, uint64_t Integer);

  /// addSInt - Add an signed integer attribute data and value.
  ///
  void addSInt(DIE *Die, unsigned Attribute, unsigned Form, int64_t Integer);

  /// addString - Add a string attribute data and value.
  ///
  void addString(DIE *Die, unsigned Attribute, const StringRef Str);

  /// addLabel - Add a Dwarf label attribute data and value.
  ///
  void addLabel(DIE *Die, unsigned Attribute, unsigned Form,
                const MCSymbol *Label);

  /// addDelta - Add a label delta attribute data and value.
  ///
  void addDelta(DIE *Die, unsigned Attribute, unsigned Form,
                const MCSymbol *Hi, const MCSymbol *Lo);

  /// addDIEEntry - Add a DIE attribute data and value.
  ///
  void addDIEEntry(DIE *Die, unsigned Attribute, unsigned Form, DIE *Entry);
  
  /// addBlock - Add block data.
  ///
  void addBlock(DIE *Die, unsigned Attribute, unsigned Form, DIEBlock *Block);

  /// addSourceLine - Add location information to specified debug information
  /// entry.
  void addSourceLine(DIE *Die, DIVariable V);
  void addSourceLine(DIE *Die, DIGlobalVariable G);
  void addSourceLine(DIE *Die, DISubprogram SP);
  void addSourceLine(DIE *Die, DIType Ty);
  void addSourceLine(DIE *Die, DINameSpace NS);
  void addSourceLine(DIE *Die, DIObjCProperty Ty);

  /// addAddress - Add an address attribute to a die based on the location
  /// provided.
  void addAddress(DIE *Die, unsigned Attribute,
                  const MachineLocation &Location);

  /// addConstantValue - Add constant value entry in variable DIE.
  bool addConstantValue(DIE *Die, const MachineOperand &MO, DIType Ty);
  bool addConstantValue(DIE *Die, const ConstantInt *CI, bool Unsigned);

  /// addConstantFPValue - Add constant value entry in variable DIE.
  bool addConstantFPValue(DIE *Die, const MachineOperand &MO);

  /// addTemplateParams - Add template parameters in buffer.
  void addTemplateParams(DIE &Buffer, DIArray TParams);

  /// addRegisterOp - Add register operand.
  void addRegisterOp(DIE *TheDie, unsigned Reg);

  /// addRegisterOffset - Add register offset.
  void addRegisterOffset(DIE *TheDie, unsigned Reg, int64_t Offset);

  /// addComplexAddress - Start with the address based on the location provided,
  /// and generate the DWARF information necessary to find the actual variable
  /// (navigating the extra location information encoded in the type) based on
  /// the starting location.  Add the DWARF information to the die.
  ///
  void addComplexAddress(DbgVariable *&DV, DIE *Die, unsigned Attribute,
                         const MachineLocation &Location);

  // FIXME: Should be reformulated in terms of addComplexAddress.
  /// addBlockByrefAddress - Start with the address based on the location
  /// provided, and generate the DWARF information necessary to find the
  /// actual Block variable (navigating the Block struct) based on the
  /// starting location.  Add the DWARF information to the die.  Obsolete,
  /// please use addComplexAddress instead.
  ///
  void addBlockByrefAddress(DbgVariable *&DV, DIE *Die, unsigned Attribute,
                            const MachineLocation &Location);

  /// addVariableAddress - Add DW_AT_location attribute for a 
  /// DbgVariable based on provided MachineLocation.
  void addVariableAddress(DbgVariable *&DV, DIE *Die, MachineLocation Location);

  /// addToContextOwner - Add Die into the list of its context owner's children.
  void addToContextOwner(DIE *Die, DIDescriptor Context);

  /// addType - Add a new type attribute to the specified entity. This takes
  /// and attribute parameter because DW_AT_friend attributes are also
  /// type references.
  void addType(DIE *Entity, DIType Ty, unsigned Attribute = dwarf::DW_AT_type);

  /// getOrCreateNameSpace - Create a DIE for DINameSpace.
  DIE *getOrCreateNameSpace(DINameSpace NS);

  /// getOrCreateSubprogramDIE - Create new DIE using SP.
  DIE *getOrCreateSubprogramDIE(DISubprogram SP);

  /// getOrCreateTypeDIE - Find existing DIE or create new DIE for the
  /// given DIType.
  DIE *getOrCreateTypeDIE(const MDNode *N);

  /// getOrCreateTemplateTypeParameterDIE - Find existing DIE or create new DIE 
  /// for the given DITemplateTypeParameter.
  DIE *getOrCreateTemplateTypeParameterDIE(DITemplateTypeParameter TP);

  /// getOrCreateTemplateValueParameterDIE - Find existing DIE or create
  /// new DIE for the given DITemplateValueParameter.
  DIE *getOrCreateTemplateValueParameterDIE(DITemplateValueParameter TVP);

  /// createDIEEntry - Creates a new DIEEntry to be a proxy for a debug
  /// information entry.
  DIEEntry *createDIEEntry(DIE *Entry);

  /// createGlobalVariableDIE - create global variable DIE.
  void createGlobalVariableDIE(const MDNode *N);

  void addPubTypes(DISubprogram SP);

  /// constructTypeDIE - Construct basic type die from DIBasicType.
  void constructTypeDIE(DIE &Buffer,
                        DIBasicType BTy);

  /// constructTypeDIE - Construct derived type die from DIDerivedType.
  void constructTypeDIE(DIE &Buffer,
                        DIDerivedType DTy);

  /// constructTypeDIE - Construct type DIE from DICompositeType.
  void constructTypeDIE(DIE &Buffer,
                        DICompositeType CTy);

  /// constructSubrangeDIE - Construct subrange DIE from DISubrange.
  void constructSubrangeDIE(DIE &Buffer, DISubrange SR, DIE *IndexTy);

  /// constructArrayTypeDIE - Construct array type DIE from DICompositeType.
  void constructArrayTypeDIE(DIE &Buffer, 
                             DICompositeType *CTy);

  /// constructEnumTypeDIE - Construct enum type DIE from DIEnumerator.
  DIE *constructEnumTypeDIE(DIEnumerator ETy);

  /// constructContainingTypeDIEs - Construct DIEs for types that contain
  /// vtables.
  void constructContainingTypeDIEs();

  /// constructVariableDIE - Construct a DIE for the given DbgVariable.
  DIE *constructVariableDIE(DbgVariable *DV, bool isScopeAbstract);

  /// createMemberDIE - Create new member DIE.
  DIE *createMemberDIE(DIDerivedType DT);

private:

  // DIEValueAllocator - All DIEValues are allocated through this allocator.
  BumpPtrAllocator DIEValueAllocator;
  DIEInteger *DIEIntegerOne;
};

} // end llvm namespace
#endif
