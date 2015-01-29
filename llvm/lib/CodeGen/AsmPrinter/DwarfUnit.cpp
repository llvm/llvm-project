//===-- llvm/CodeGen/DwarfUnit.cpp - Dwarf Type and Compile Units ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains support for constructing a dwarf compile unit.
//
//===----------------------------------------------------------------------===//

#include "DwarfUnit.h"
#include "DwarfAccelTable.h"
#include "DwarfCompileUnit.h"
#include "DwarfDebug.h"
#include "DwarfExpression.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Mangler.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCSection.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Target/TargetFrameLowering.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Target/TargetSubtargetInfo.h"

using namespace llvm;

#define DEBUG_TYPE "dwarfdebug"

static cl::opt<bool>
GenerateDwarfTypeUnits("generate-type-units", cl::Hidden,
                       cl::desc("Generate DWARF4 type units."),
                       cl::init(false));

void DIEDwarfExpression::EmitOp(uint8_t Op, const char* Comment) {
  DU.addUInt(DIE, dwarf::DW_FORM_data1, Op);
}
void DIEDwarfExpression::EmitSigned(int Value) {
  DU.addSInt(DIE, dwarf::DW_FORM_sdata, Value);
}
void DIEDwarfExpression::EmitUnsigned(unsigned Value) {
  DU.addUInt(DIE, dwarf::DW_FORM_udata, Value);
}
bool DIEDwarfExpression::isFrameRegister(unsigned MachineReg) {
  return MachineReg == getTRI()->getFrameRegister(*AP.MF);
}


/// Unit - Unit constructor.
DwarfUnit::DwarfUnit(unsigned UID, dwarf::Tag UnitTag, DICompileUnit Node,
                     AsmPrinter *A, DwarfDebug *DW, DwarfFile *DWU)
    : UniqueID(UID), CUNode(Node), UnitDie(UnitTag), DebugInfoOffset(0), Asm(A),
      DD(DW), DU(DWU), IndexTyDie(nullptr), Section(nullptr) {
  assert(UnitTag == dwarf::DW_TAG_compile_unit ||
         UnitTag == dwarf::DW_TAG_type_unit);
  DIEIntegerOne = new (DIEValueAllocator) DIEInteger(1);
}

DwarfTypeUnit::DwarfTypeUnit(unsigned UID, DwarfCompileUnit &CU, AsmPrinter *A,
                             DwarfDebug *DW, DwarfFile *DWU,
                             MCDwarfDwoLineTable *SplitLineTable)
    : DwarfUnit(UID, dwarf::DW_TAG_type_unit, CU.getCUNode(), A, DW, DWU),
      CU(CU), SplitLineTable(SplitLineTable) {
  if (SplitLineTable)
    addSectionOffset(UnitDie, dwarf::DW_AT_stmt_list, 0);
}

/// ~Unit - Destructor for compile unit.
DwarfUnit::~DwarfUnit() {
  for (unsigned j = 0, M = DIEBlocks.size(); j < M; ++j)
    DIEBlocks[j]->~DIEBlock();
  for (unsigned j = 0, M = DIELocs.size(); j < M; ++j)
    DIELocs[j]->~DIELoc();
}

/// createDIEEntry - Creates a new DIEEntry to be a proxy for a debug
/// information entry.
DIEEntry *DwarfUnit::createDIEEntry(DIE &Entry) {
  DIEEntry *Value = new (DIEValueAllocator) DIEEntry(Entry);
  return Value;
}

/// getDefaultLowerBound - Return the default lower bound for an array. If the
/// DWARF version doesn't handle the language, return -1.
int64_t DwarfUnit::getDefaultLowerBound() const {
  switch (getLanguage()) {
  default:
    break;

  case dwarf::DW_LANG_C89:
  case dwarf::DW_LANG_C99:
  case dwarf::DW_LANG_C:
  case dwarf::DW_LANG_C_plus_plus:
  case dwarf::DW_LANG_ObjC:
  case dwarf::DW_LANG_ObjC_plus_plus:
    return 0;

  case dwarf::DW_LANG_Fortran77:
  case dwarf::DW_LANG_Fortran90:
  case dwarf::DW_LANG_Fortran95:
    return 1;

  // The languages below have valid values only if the DWARF version >= 4.
  case dwarf::DW_LANG_Java:
  case dwarf::DW_LANG_Python:
  case dwarf::DW_LANG_UPC:
  case dwarf::DW_LANG_D:
    if (dwarf::DWARF_VERSION >= 4)
      return 0;
    break;

  case dwarf::DW_LANG_Ada83:
  case dwarf::DW_LANG_Ada95:
  case dwarf::DW_LANG_Cobol74:
  case dwarf::DW_LANG_Cobol85:
  case dwarf::DW_LANG_Modula2:
  case dwarf::DW_LANG_Pascal83:
  case dwarf::DW_LANG_PLI:
    if (dwarf::DWARF_VERSION >= 4)
      return 1;
    break;
  }

  return -1;
}

/// Check whether the DIE for this MDNode can be shared across CUs.
static bool isShareableAcrossCUs(DIDescriptor D) {
  // When the MDNode can be part of the type system, the DIE can be shared
  // across CUs.
  // Combining type units and cross-CU DIE sharing is lower value (since
  // cross-CU DIE sharing is used in LTO and removes type redundancy at that
  // level already) but may be implementable for some value in projects
  // building multiple independent libraries with LTO and then linking those
  // together.
  return (D.isType() ||
          (D.isSubprogram() && !DISubprogram(D).isDefinition())) &&
         !GenerateDwarfTypeUnits;
}

/// getDIE - Returns the debug information entry map slot for the
/// specified debug variable. We delegate the request to DwarfDebug
/// when the DIE for this MDNode can be shared across CUs. The mappings
/// will be kept in DwarfDebug for shareable DIEs.
DIE *DwarfUnit::getDIE(DIDescriptor D) const {
  if (isShareableAcrossCUs(D))
    return DU->getDIE(D);
  return MDNodeToDieMap.lookup(D);
}

/// insertDIE - Insert DIE into the map. We delegate the request to DwarfDebug
/// when the DIE for this MDNode can be shared across CUs. The mappings
/// will be kept in DwarfDebug for shareable DIEs.
void DwarfUnit::insertDIE(DIDescriptor Desc, DIE *D) {
  if (isShareableAcrossCUs(Desc)) {
    DU->insertDIE(Desc, D);
    return;
  }
  MDNodeToDieMap.insert(std::make_pair(Desc, D));
}

/// addFlag - Add a flag that is true.
void DwarfUnit::addFlag(DIE &Die, dwarf::Attribute Attribute) {
  if (DD->getDwarfVersion() >= 4)
    Die.addValue(Attribute, dwarf::DW_FORM_flag_present, DIEIntegerOne);
  else
    Die.addValue(Attribute, dwarf::DW_FORM_flag, DIEIntegerOne);
}

/// addUInt - Add an unsigned integer attribute data and value.
///
void DwarfUnit::addUInt(DIE &Die, dwarf::Attribute Attribute,
                        Optional<dwarf::Form> Form, uint64_t Integer) {
  if (!Form)
    Form = DIEInteger::BestForm(false, Integer);
  DIEValue *Value = Integer == 1 ? DIEIntegerOne : new (DIEValueAllocator)
                        DIEInteger(Integer);
  Die.addValue(Attribute, *Form, Value);
}

void DwarfUnit::addUInt(DIE &Block, dwarf::Form Form, uint64_t Integer) {
  addUInt(Block, (dwarf::Attribute)0, Form, Integer);
}

/// addSInt - Add an signed integer attribute data and value.
///
void DwarfUnit::addSInt(DIE &Die, dwarf::Attribute Attribute,
                        Optional<dwarf::Form> Form, int64_t Integer) {
  if (!Form)
    Form = DIEInteger::BestForm(true, Integer);
  DIEValue *Value = new (DIEValueAllocator) DIEInteger(Integer);
  Die.addValue(Attribute, *Form, Value);
}

void DwarfUnit::addSInt(DIELoc &Die, Optional<dwarf::Form> Form,
                        int64_t Integer) {
  addSInt(Die, (dwarf::Attribute)0, Form, Integer);
}

/// addString - Add a string attribute data and value. We always emit a
/// reference to the string pool instead of immediate strings so that DIEs have
/// more predictable sizes. In the case of split dwarf we emit an index
/// into another table which gets us the static offset into the string
/// table.
void DwarfUnit::addString(DIE &Die, dwarf::Attribute Attribute,
                          StringRef String) {
  if (!isDwoUnit())
    return addLocalString(Die, Attribute, String);

  addIndexedString(Die, Attribute, String);
}

void DwarfUnit::addIndexedString(DIE &Die, dwarf::Attribute Attribute,
                                 StringRef String) {
  unsigned idx = DU->getStringPool().getIndex(*Asm, String);
  DIEValue *Value = new (DIEValueAllocator) DIEInteger(idx);
  DIEValue *Str = new (DIEValueAllocator) DIEString(Value, String);
  Die.addValue(Attribute, dwarf::DW_FORM_GNU_str_index, Str);
}

/// addLocalString - Add a string attribute data and value. This is guaranteed
/// to be in the local string pool instead of indirected.
void DwarfUnit::addLocalString(DIE &Die, dwarf::Attribute Attribute,
                               StringRef String) {
  MCSymbol *Symb = DU->getStringPool().getSymbol(*Asm, String);
  DIEValue *Value;
  if (Asm->MAI->doesDwarfUseRelocationsAcrossSections())
    Value = new (DIEValueAllocator) DIELabel(Symb);
  else
    Value = new (DIEValueAllocator) DIEDelta(Symb, DD->getDebugStrSym());
  DIEValue *Str = new (DIEValueAllocator) DIEString(Value, String);
  Die.addValue(Attribute, dwarf::DW_FORM_strp, Str);
}

/// addLabel - Add a Dwarf label attribute data and value.
///
void DwarfUnit::addLabel(DIE &Die, dwarf::Attribute Attribute, dwarf::Form Form,
                         const MCSymbol *Label) {
  DIEValue *Value = new (DIEValueAllocator) DIELabel(Label);
  Die.addValue(Attribute, Form, Value);
}

void DwarfUnit::addLabel(DIELoc &Die, dwarf::Form Form, const MCSymbol *Label) {
  addLabel(Die, (dwarf::Attribute)0, Form, Label);
}

/// addSectionOffset - Add an offset into a section attribute data and value.
///
void DwarfUnit::addSectionOffset(DIE &Die, dwarf::Attribute Attribute,
                                 uint64_t Integer) {
  if (DD->getDwarfVersion() >= 4)
    addUInt(Die, Attribute, dwarf::DW_FORM_sec_offset, Integer);
  else
    addUInt(Die, Attribute, dwarf::DW_FORM_data4, Integer);
}

unsigned DwarfTypeUnit::getOrCreateSourceID(StringRef FileName, StringRef DirName) {
  return SplitLineTable ? SplitLineTable->getFile(DirName, FileName)
                        : getCU().getOrCreateSourceID(FileName, DirName);
}

/// addOpAddress - Add a dwarf op address data and value using the
/// form given and an op of either DW_FORM_addr or DW_FORM_GNU_addr_index.
///
void DwarfUnit::addOpAddress(DIELoc &Die, const MCSymbol *Sym) {
  if (!DD->useSplitDwarf()) {
    addUInt(Die, dwarf::DW_FORM_data1, dwarf::DW_OP_addr);
    addLabel(Die, dwarf::DW_FORM_udata, Sym);
  } else {
    addUInt(Die, dwarf::DW_FORM_data1, dwarf::DW_OP_GNU_addr_index);
    addUInt(Die, dwarf::DW_FORM_GNU_addr_index,
            DD->getAddressPool().getIndex(Sym));
  }
}

void DwarfUnit::addLabelDelta(DIE &Die, dwarf::Attribute Attribute,
                              const MCSymbol *Hi, const MCSymbol *Lo) {
  DIEValue *Value = new (DIEValueAllocator) DIEDelta(Hi, Lo);
  Die.addValue(Attribute, dwarf::DW_FORM_data4, Value);
}

/// addDIEEntry - Add a DIE attribute data and value.
///
void DwarfUnit::addDIEEntry(DIE &Die, dwarf::Attribute Attribute, DIE &Entry) {
  addDIEEntry(Die, Attribute, createDIEEntry(Entry));
}

void DwarfUnit::addDIETypeSignature(DIE &Die, const DwarfTypeUnit &Type) {
  // Flag the type unit reference as a declaration so that if it contains
  // members (implicit special members, static data member definitions, member
  // declarations for definitions in this CU, etc) consumers don't get confused
  // and think this is a full definition.
  addFlag(Die, dwarf::DW_AT_declaration);

  Die.addValue(dwarf::DW_AT_signature, dwarf::DW_FORM_ref_sig8,
               new (DIEValueAllocator) DIETypeSignature(Type));
}

void DwarfUnit::addDIEEntry(DIE &Die, dwarf::Attribute Attribute,
                            DIEEntry *Entry) {
  const DIE *DieCU = Die.getUnitOrNull();
  const DIE *EntryCU = Entry->getEntry().getUnitOrNull();
  if (!DieCU)
    // We assume that Die belongs to this CU, if it is not linked to any CU yet.
    DieCU = &getUnitDie();
  if (!EntryCU)
    EntryCU = &getUnitDie();
  Die.addValue(Attribute,
               EntryCU == DieCU ? dwarf::DW_FORM_ref4 : dwarf::DW_FORM_ref_addr,
               Entry);
}

/// Create a DIE with the given Tag, add the DIE to its parent, and
/// call insertDIE if MD is not null.
DIE &DwarfUnit::createAndAddDIE(unsigned Tag, DIE &Parent, DIDescriptor N) {
  assert(Tag != dwarf::DW_TAG_auto_variable &&
         Tag != dwarf::DW_TAG_arg_variable);
  Parent.addChild(make_unique<DIE>((dwarf::Tag)Tag));
  DIE &Die = *Parent.getChildren().back();
  if (N)
    insertDIE(N, &Die);
  return Die;
}

/// addBlock - Add block data.
///
void DwarfUnit::addBlock(DIE &Die, dwarf::Attribute Attribute, DIELoc *Loc) {
  Loc->ComputeSize(Asm);
  DIELocs.push_back(Loc); // Memoize so we can call the destructor later on.
  Die.addValue(Attribute, Loc->BestForm(DD->getDwarfVersion()), Loc);
}

void DwarfUnit::addBlock(DIE &Die, dwarf::Attribute Attribute,
                         DIEBlock *Block) {
  Block->ComputeSize(Asm);
  DIEBlocks.push_back(Block); // Memoize so we can call the destructor later on.
  Die.addValue(Attribute, Block->BestForm(), Block);
}

/// addSourceLine - Add location information to specified debug information
/// entry.
void DwarfUnit::addSourceLine(DIE &Die, unsigned Line, StringRef File,
                              StringRef Directory) {
  if (Line == 0)
    return;

  unsigned FileID = getOrCreateSourceID(File, Directory);
  assert(FileID && "Invalid file id");
  addUInt(Die, dwarf::DW_AT_decl_file, None, FileID);
  addUInt(Die, dwarf::DW_AT_decl_line, None, Line);
}

/// addSourceLine - Add location information to specified debug information
/// entry.
void DwarfUnit::addSourceLine(DIE &Die, DIVariable V) {
  assert(V.isVariable());

  addSourceLine(Die, V.getLineNumber(), V.getContext().getFilename(),
                V.getContext().getDirectory());
}

/// addSourceLine - Add location information to specified debug information
/// entry.
void DwarfUnit::addSourceLine(DIE &Die, DIGlobalVariable G) {
  assert(G.isGlobalVariable());

  addSourceLine(Die, G.getLineNumber(), G.getFilename(), G.getDirectory());
}

/// addSourceLine - Add location information to specified debug information
/// entry.
void DwarfUnit::addSourceLine(DIE &Die, DISubprogram SP) {
  assert(SP.isSubprogram());

  addSourceLine(Die, SP.getLineNumber(), SP.getFilename(), SP.getDirectory());
}

/// addSourceLine - Add location information to specified debug information
/// entry.
void DwarfUnit::addSourceLine(DIE &Die, DIType Ty) {
  assert(Ty.isType());

  addSourceLine(Die, Ty.getLineNumber(), Ty.getFilename(), Ty.getDirectory());
}

/// addSourceLine - Add location information to specified debug information
/// entry.
void DwarfUnit::addSourceLine(DIE &Die, DIObjCProperty Ty) {
  assert(Ty.isObjCProperty());

  DIFile File = Ty.getFile();
  addSourceLine(Die, Ty.getLineNumber(), File.getFilename(),
                File.getDirectory());
}

/// addSourceLine - Add location information to specified debug information
/// entry.
void DwarfUnit::addSourceLine(DIE &Die, DINameSpace NS) {
  assert(NS.Verify());

  addSourceLine(Die, NS.getLineNumber(), NS.getFilename(), NS.getDirectory());
}

/// addRegisterOp - Add register operand.
bool DwarfUnit::addRegisterOpPiece(DIELoc &TheDie, unsigned Reg,
                                   unsigned SizeInBits, unsigned OffsetInBits) {
  DIEDwarfExpression Expr(*Asm, *this, TheDie);
  Expr.AddMachineRegPiece(Reg, SizeInBits, OffsetInBits);
  return true;
}

/// addRegisterOffset - Add register offset.
bool DwarfUnit::addRegisterOffset(DIELoc &TheDie, unsigned Reg,
                                  int64_t Offset) {
  DIEDwarfExpression Expr(*Asm, *this, TheDie);
  return Expr.AddMachineRegIndirect(Reg, Offset);
}

/* Byref variables, in Blocks, are declared by the programmer as "SomeType
   VarName;", but the compiler creates a __Block_byref_x_VarName struct, and
   gives the variable VarName either the struct, or a pointer to the struct, as
   its type.  This is necessary for various behind-the-scenes things the
   compiler needs to do with by-reference variables in Blocks.

   However, as far as the original *programmer* is concerned, the variable
   should still have type 'SomeType', as originally declared.

   The function getBlockByrefType dives into the __Block_byref_x_VarName
   struct to find the original type of the variable, which is then assigned to
   the variable's Debug Information Entry as its real type.  So far, so good.
   However now the debugger will expect the variable VarName to have the type
   SomeType.  So we need the location attribute for the variable to be an
   expression that explains to the debugger how to navigate through the
   pointers and struct to find the actual variable of type SomeType.

   The following function does just that.  We start by getting
   the "normal" location for the variable. This will be the location
   of either the struct __Block_byref_x_VarName or the pointer to the
   struct __Block_byref_x_VarName.

   The struct will look something like:

   struct __Block_byref_x_VarName {
     ... <various fields>
     struct __Block_byref_x_VarName *forwarding;
     ... <various other fields>
     SomeType VarName;
     ... <maybe more fields>
   };

   If we are given the struct directly (as our starting point) we
   need to tell the debugger to:

   1).  Add the offset of the forwarding field.

   2).  Follow that pointer to get the real __Block_byref_x_VarName
   struct to use (the real one may have been copied onto the heap).

   3).  Add the offset for the field VarName, to find the actual variable.

   If we started with a pointer to the struct, then we need to
   dereference that pointer first, before the other steps.
   Translating this into DWARF ops, we will need to append the following
   to the current location description for the variable:

   DW_OP_deref                    -- optional, if we start with a pointer
   DW_OP_plus_uconst <forward_fld_offset>
   DW_OP_deref
   DW_OP_plus_uconst <varName_fld_offset>

   That is what this function does.  */

/// addBlockByrefAddress - Start with the address based on the location
/// provided, and generate the DWARF information necessary to find the
/// actual Block variable (navigating the Block struct) based on the
/// starting location.  Add the DWARF information to the die.  For
/// more information, read large comment just above here.
///
void DwarfUnit::addBlockByrefAddress(const DbgVariable &DV, DIE &Die,
                                     dwarf::Attribute Attribute,
                                     const MachineLocation &Location) {
  DIType Ty = DV.getType();
  DIType TmpTy = Ty;
  uint16_t Tag = Ty.getTag();
  bool isPointer = false;

  StringRef varName = DV.getName();

  if (Tag == dwarf::DW_TAG_pointer_type) {
    DIDerivedType DTy(Ty);
    TmpTy = resolve(DTy.getTypeDerivedFrom());
    isPointer = true;
  }

  DICompositeType blockStruct(TmpTy);

  // Find the __forwarding field and the variable field in the __Block_byref
  // struct.
  DIArray Fields = blockStruct.getElements();
  DIDerivedType varField;
  DIDerivedType forwardingField;

  for (unsigned i = 0, N = Fields.getNumElements(); i < N; ++i) {
    DIDerivedType DT(Fields.getElement(i));
    StringRef fieldName = DT.getName();
    if (fieldName == "__forwarding")
      forwardingField = DT;
    else if (fieldName == varName)
      varField = DT;
  }

  // Get the offsets for the forwarding field and the variable field.
  unsigned forwardingFieldOffset = forwardingField.getOffsetInBits() >> 3;
  unsigned varFieldOffset = varField.getOffsetInBits() >> 2;

  // Decode the original location, and use that as the start of the byref
  // variable's location.
  DIELoc *Loc = new (DIEValueAllocator) DIELoc();

  bool validReg;
  if (Location.isReg())
    validReg = addRegisterOpPiece(*Loc, Location.getReg());
  else
    validReg = addRegisterOffset(*Loc, Location.getReg(), Location.getOffset());

  if (!validReg)
    return;

  // If we started with a pointer to the __Block_byref... struct, then
  // the first thing we need to do is dereference the pointer (DW_OP_deref).
  if (isPointer)
    addUInt(*Loc, dwarf::DW_FORM_data1, dwarf::DW_OP_deref);

  // Next add the offset for the '__forwarding' field:
  // DW_OP_plus_uconst ForwardingFieldOffset.  Note there's no point in
  // adding the offset if it's 0.
  if (forwardingFieldOffset > 0) {
    addUInt(*Loc, dwarf::DW_FORM_data1, dwarf::DW_OP_plus_uconst);
    addUInt(*Loc, dwarf::DW_FORM_udata, forwardingFieldOffset);
  }

  // Now dereference the __forwarding field to get to the real __Block_byref
  // struct:  DW_OP_deref.
  addUInt(*Loc, dwarf::DW_FORM_data1, dwarf::DW_OP_deref);

  // Now that we've got the real __Block_byref... struct, add the offset
  // for the variable's field to get to the location of the actual variable:
  // DW_OP_plus_uconst varFieldOffset.  Again, don't add if it's 0.
  if (varFieldOffset > 0) {
    addUInt(*Loc, dwarf::DW_FORM_data1, dwarf::DW_OP_plus_uconst);
    addUInt(*Loc, dwarf::DW_FORM_udata, varFieldOffset);
  }

  // Now attach the location information to the DIE.
  addBlock(Die, Attribute, Loc);
}

/// Return true if type encoding is unsigned.
static bool isUnsignedDIType(DwarfDebug *DD, DIType Ty) {
  DIDerivedType DTy(Ty);
  if (DTy.isDerivedType()) {
    dwarf::Tag T = (dwarf::Tag)Ty.getTag();
    // Encode pointer constants as unsigned bytes. This is used at least for
    // null pointer constant emission.
    // (Pieces of) aggregate types that get hacked apart by SROA may also be
    // represented by a constant. Encode them as unsigned bytes.
    // FIXME: reference and rvalue_reference /probably/ shouldn't be allowed
    // here, but accept them for now due to a bug in SROA producing bogus
    // dbg.values.
    if (T == dwarf::DW_TAG_array_type ||
        T == dwarf::DW_TAG_class_type ||
        T == dwarf::DW_TAG_pointer_type ||
        T == dwarf::DW_TAG_ptr_to_member_type ||
        T == dwarf::DW_TAG_reference_type ||
        T == dwarf::DW_TAG_rvalue_reference_type ||
        T == dwarf::DW_TAG_structure_type)
      return true;
    assert(T == dwarf::DW_TAG_typedef || T == dwarf::DW_TAG_const_type ||
           T == dwarf::DW_TAG_volatile_type ||
           T == dwarf::DW_TAG_restrict_type ||
           T == dwarf::DW_TAG_enumeration_type);
    if (DITypeRef Deriv = DTy.getTypeDerivedFrom())
      return isUnsignedDIType(DD, DD->resolve(Deriv));
    // FIXME: Enums without a fixed underlying type have unknown signedness
    // here, leading to incorrectly emitted constants.
    assert(DTy.getTag() == dwarf::DW_TAG_enumeration_type);
    return false;
  }

  DIBasicType BTy(Ty);
  assert(BTy.isBasicType());
  unsigned Encoding = BTy.getEncoding();
  assert((Encoding == dwarf::DW_ATE_unsigned ||
          Encoding == dwarf::DW_ATE_unsigned_char ||
          Encoding == dwarf::DW_ATE_signed ||
          Encoding == dwarf::DW_ATE_signed_char ||
          Encoding == dwarf::DW_ATE_UTF || Encoding == dwarf::DW_ATE_boolean ||
          (Ty.getTag() == dwarf::DW_TAG_unspecified_type &&
           Ty.getName() == "decltype(nullptr)")) &&
         "Unsupported encoding");
  return (Encoding == dwarf::DW_ATE_unsigned ||
          Encoding == dwarf::DW_ATE_unsigned_char ||
          Encoding == dwarf::DW_ATE_UTF || Encoding == dwarf::DW_ATE_boolean ||
          Ty.getTag() == dwarf::DW_TAG_unspecified_type);
}

/// If this type is derived from a base type then return base type size.
static uint64_t getBaseTypeSize(DwarfDebug *DD, DIDerivedType Ty) {
  unsigned Tag = Ty.getTag();

  if (Tag != dwarf::DW_TAG_member && Tag != dwarf::DW_TAG_typedef &&
      Tag != dwarf::DW_TAG_const_type && Tag != dwarf::DW_TAG_volatile_type &&
      Tag != dwarf::DW_TAG_restrict_type)
    return Ty.getSizeInBits();

  DIType BaseType = DD->resolve(Ty.getTypeDerivedFrom());

  assert(BaseType.isValid());

  // If this is a derived type, go ahead and get the base type, unless it's a
  // reference then it's just the size of the field. Pointer types have no need
  // of this since they're a different type of qualification on the type.
  if (BaseType.getTag() == dwarf::DW_TAG_reference_type ||
      BaseType.getTag() == dwarf::DW_TAG_rvalue_reference_type)
    return Ty.getSizeInBits();

  if (BaseType.isDerivedType())
    return getBaseTypeSize(DD, DIDerivedType(BaseType));

  return BaseType.getSizeInBits();
}

/// addConstantFPValue - Add constant value entry in variable DIE.
void DwarfUnit::addConstantFPValue(DIE &Die, const MachineOperand &MO) {
  assert(MO.isFPImm() && "Invalid machine operand!");
  DIEBlock *Block = new (DIEValueAllocator) DIEBlock();
  APFloat FPImm = MO.getFPImm()->getValueAPF();

  // Get the raw data form of the floating point.
  const APInt FltVal = FPImm.bitcastToAPInt();
  const char *FltPtr = (const char *)FltVal.getRawData();

  int NumBytes = FltVal.getBitWidth() / 8; // 8 bits per byte.
  bool LittleEndian = Asm->getDataLayout().isLittleEndian();
  int Incr = (LittleEndian ? 1 : -1);
  int Start = (LittleEndian ? 0 : NumBytes - 1);
  int Stop = (LittleEndian ? NumBytes : -1);

  // Output the constant to DWARF one byte at a time.
  for (; Start != Stop; Start += Incr)
    addUInt(*Block, dwarf::DW_FORM_data1, (unsigned char)0xFF & FltPtr[Start]);

  addBlock(Die, dwarf::DW_AT_const_value, Block);
}

/// addConstantFPValue - Add constant value entry in variable DIE.
void DwarfUnit::addConstantFPValue(DIE &Die, const ConstantFP *CFP) {
  // Pass this down to addConstantValue as an unsigned bag of bits.
  addConstantValue(Die, CFP->getValueAPF().bitcastToAPInt(), true);
}

/// addConstantValue - Add constant value entry in variable DIE.
void DwarfUnit::addConstantValue(DIE &Die, const ConstantInt *CI, DIType Ty) {
  addConstantValue(Die, CI->getValue(), Ty);
}

/// addConstantValue - Add constant value entry in variable DIE.
void DwarfUnit::addConstantValue(DIE &Die, const MachineOperand &MO,
                                 DIType Ty) {
  assert(MO.isImm() && "Invalid machine operand!");

  addConstantValue(Die, isUnsignedDIType(DD, Ty), MO.getImm());
}

void DwarfUnit::addConstantValue(DIE &Die, bool Unsigned, uint64_t Val) {
  // FIXME: This is a bit conservative/simple - it emits negative values always
  // sign extended to 64 bits rather than minimizing the number of bytes.
  addUInt(Die, dwarf::DW_AT_const_value,
          Unsigned ? dwarf::DW_FORM_udata : dwarf::DW_FORM_sdata, Val);
}

void DwarfUnit::addConstantValue(DIE &Die, const APInt &Val, DIType Ty) {
  addConstantValue(Die, Val, isUnsignedDIType(DD, Ty));
}

// addConstantValue - Add constant value entry in variable DIE.
void DwarfUnit::addConstantValue(DIE &Die, const APInt &Val, bool Unsigned) {
  unsigned CIBitWidth = Val.getBitWidth();
  if (CIBitWidth <= 64) {
    addConstantValue(Die, Unsigned,
                     Unsigned ? Val.getZExtValue() : Val.getSExtValue());
    return;
  }

  DIEBlock *Block = new (DIEValueAllocator) DIEBlock();

  // Get the raw data form of the large APInt.
  const uint64_t *Ptr64 = Val.getRawData();

  int NumBytes = Val.getBitWidth() / 8; // 8 bits per byte.
  bool LittleEndian = Asm->getDataLayout().isLittleEndian();

  // Output the constant to DWARF one byte at a time.
  for (int i = 0; i < NumBytes; i++) {
    uint8_t c;
    if (LittleEndian)
      c = Ptr64[i / 8] >> (8 * (i & 7));
    else
      c = Ptr64[(NumBytes - 1 - i) / 8] >> (8 * ((NumBytes - 1 - i) & 7));
    addUInt(*Block, dwarf::DW_FORM_data1, c);
  }

  addBlock(Die, dwarf::DW_AT_const_value, Block);
}

/// addTemplateParams - Add template parameters into buffer.
void DwarfUnit::addTemplateParams(DIE &Buffer, DIArray TParams) {
  // Add template parameters.
  for (unsigned i = 0, e = TParams.getNumElements(); i != e; ++i) {
    DIDescriptor Element = TParams.getElement(i);
    if (Element.isTemplateTypeParameter())
      constructTemplateTypeParameterDIE(Buffer,
                                        DITemplateTypeParameter(Element));
    else if (Element.isTemplateValueParameter())
      constructTemplateValueParameterDIE(Buffer,
                                         DITemplateValueParameter(Element));
  }
}

/// getOrCreateContextDIE - Get context owner's DIE.
DIE *DwarfUnit::getOrCreateContextDIE(DIScope Context) {
  if (!Context || Context.isFile())
    return &getUnitDie();
  if (Context.isType())
    return getOrCreateTypeDIE(DIType(Context));
  if (Context.isNameSpace())
    return getOrCreateNameSpace(DINameSpace(Context));
  if (Context.isSubprogram())
    return getOrCreateSubprogramDIE(DISubprogram(Context));
  return getDIE(Context);
}

DIE *DwarfUnit::createTypeDIE(DICompositeType Ty) {
  DIScope Context = resolve(Ty.getContext());
  DIE *ContextDIE = getOrCreateContextDIE(Context);

  if (DIE *TyDIE = getDIE(Ty))
    return TyDIE;

  // Create new type.
  DIE &TyDIE = createAndAddDIE(Ty.getTag(), *ContextDIE, Ty);

  constructTypeDIE(TyDIE, Ty);

  updateAcceleratorTables(Context, Ty, TyDIE);
  return &TyDIE;
}

/// getOrCreateTypeDIE - Find existing DIE or create new DIE for the
/// given DIType.
DIE *DwarfUnit::getOrCreateTypeDIE(const MDNode *TyNode) {
  if (!TyNode)
    return nullptr;

  DIType Ty(TyNode);
  assert(Ty.isType());
  assert(Ty == resolve(Ty.getRef()) &&
         "type was not uniqued, possible ODR violation.");

  // DW_TAG_restrict_type is not supported in DWARF2
  if (Ty.getTag() == dwarf::DW_TAG_restrict_type && DD->getDwarfVersion() <= 2)
    return getOrCreateTypeDIE(resolve(DIDerivedType(Ty).getTypeDerivedFrom()));

  // Construct the context before querying for the existence of the DIE in case
  // such construction creates the DIE.
  DIScope Context = resolve(Ty.getContext());
  DIE *ContextDIE = getOrCreateContextDIE(Context);
  assert(ContextDIE);

  if (DIE *TyDIE = getDIE(Ty))
    return TyDIE;

  // Create new type.
  DIE &TyDIE = createAndAddDIE(Ty.getTag(), *ContextDIE, Ty);

  updateAcceleratorTables(Context, Ty, TyDIE);

  if (Ty.isBasicType())
    constructTypeDIE(TyDIE, DIBasicType(Ty));
  else if (Ty.isCompositeType()) {
    DICompositeType CTy(Ty);
    if (GenerateDwarfTypeUnits && !Ty.isForwardDecl())
      if (MDString *TypeId = CTy.getIdentifier()) {
        DD->addDwarfTypeUnitType(getCU(), TypeId->getString(), TyDIE, CTy);
        // Skip updating the accelerator tables since this is not the full type.
        return &TyDIE;
      }
    constructTypeDIE(TyDIE, CTy);
  } else {
    assert(Ty.isDerivedType() && "Unknown kind of DIType");
    constructTypeDIE(TyDIE, DIDerivedType(Ty));
  }

  return &TyDIE;
}

void DwarfUnit::updateAcceleratorTables(DIScope Context, DIType Ty,
                                        const DIE &TyDIE) {
  if (!Ty.getName().empty() && !Ty.isForwardDecl()) {
    bool IsImplementation = 0;
    if (Ty.isCompositeType()) {
      DICompositeType CT(Ty);
      // A runtime language of 0 actually means C/C++ and that any
      // non-negative value is some version of Objective-C/C++.
      IsImplementation = (CT.getRunTimeLang() == 0) || CT.isObjcClassComplete();
    }
    unsigned Flags = IsImplementation ? dwarf::DW_FLAG_type_implementation : 0;
    DD->addAccelType(Ty.getName(), TyDIE, Flags);

    if (!Context || Context.isCompileUnit() || Context.isFile() ||
        Context.isNameSpace())
      addGlobalType(Ty, TyDIE, Context);
  }
}

/// addType - Add a new type attribute to the specified entity.
void DwarfUnit::addType(DIE &Entity, DIType Ty, dwarf::Attribute Attribute) {
  assert(Ty && "Trying to add a type that doesn't exist?");

  // Check for pre-existence.
  DIEEntry *Entry = getDIEEntry(Ty);
  // If it exists then use the existing value.
  if (Entry) {
    addDIEEntry(Entity, Attribute, Entry);
    return;
  }

  // Construct type.
  DIE *Buffer = getOrCreateTypeDIE(Ty);

  // Set up proxy.
  Entry = createDIEEntry(*Buffer);
  insertDIEEntry(Ty, Entry);
  addDIEEntry(Entity, Attribute, Entry);
}

/// getParentContextString - Walks the metadata parent chain in a language
/// specific manner (using the compile unit language) and returns
/// it as a string. This is done at the metadata level because DIEs may
/// not currently have been added to the parent context and walking the
/// DIEs looking for names is more expensive than walking the metadata.
std::string DwarfUnit::getParentContextString(DIScope Context) const {
  if (!Context)
    return "";

  // FIXME: Decide whether to implement this for non-C++ languages.
  if (getLanguage() != dwarf::DW_LANG_C_plus_plus)
    return "";

  std::string CS;
  SmallVector<DIScope, 1> Parents;
  while (!Context.isCompileUnit()) {
    Parents.push_back(Context);
    if (Context.getContext())
      Context = resolve(Context.getContext());
    else
      // Structure, etc types will have a NULL context if they're at the top
      // level.
      break;
  }

  // Reverse iterate over our list to go from the outermost construct to the
  // innermost.
  for (SmallVectorImpl<DIScope>::reverse_iterator I = Parents.rbegin(),
                                                  E = Parents.rend();
       I != E; ++I) {
    DIScope Ctx = *I;
    StringRef Name = Ctx.getName();
    if (Name.empty() && Ctx.isNameSpace())
      Name = "(anonymous namespace)";
    if (!Name.empty()) {
      CS += Name;
      CS += "::";
    }
  }
  return CS;
}

/// constructTypeDIE - Construct basic type die from DIBasicType.
void DwarfUnit::constructTypeDIE(DIE &Buffer, DIBasicType BTy) {
  // Get core information.
  StringRef Name = BTy.getName();
  // Add name if not anonymous or intermediate type.
  if (!Name.empty())
    addString(Buffer, dwarf::DW_AT_name, Name);

  // An unspecified type only has a name attribute.
  if (BTy.getTag() == dwarf::DW_TAG_unspecified_type)
    return;

  addUInt(Buffer, dwarf::DW_AT_encoding, dwarf::DW_FORM_data1,
          BTy.getEncoding());

  uint64_t Size = BTy.getSizeInBits() >> 3;
  addUInt(Buffer, dwarf::DW_AT_byte_size, None, Size);
}

/// constructTypeDIE - Construct derived type die from DIDerivedType.
void DwarfUnit::constructTypeDIE(DIE &Buffer, DIDerivedType DTy) {
  // Get core information.
  StringRef Name = DTy.getName();
  uint64_t Size = DTy.getSizeInBits() >> 3;
  uint16_t Tag = Buffer.getTag();

  // Map to main type, void will not have a type.
  DIType FromTy = resolve(DTy.getTypeDerivedFrom());
  if (FromTy)
    addType(Buffer, FromTy);

  // Add name if not anonymous or intermediate type.
  if (!Name.empty())
    addString(Buffer, dwarf::DW_AT_name, Name);

  // Add size if non-zero (derived types might be zero-sized.)
  if (Size && Tag != dwarf::DW_TAG_pointer_type
           && Tag != dwarf::DW_TAG_ptr_to_member_type)
    addUInt(Buffer, dwarf::DW_AT_byte_size, None, Size);

  if (Tag == dwarf::DW_TAG_ptr_to_member_type)
    addDIEEntry(Buffer, dwarf::DW_AT_containing_type,
                *getOrCreateTypeDIE(resolve(DTy.getClassType())));
  // Add source line info if available and TyDesc is not a forward declaration.
  if (!DTy.isForwardDecl())
    addSourceLine(Buffer, DTy);
}

/// constructSubprogramArguments - Construct function argument DIEs.
void DwarfUnit::constructSubprogramArguments(DIE &Buffer, DITypeArray Args) {
  for (unsigned i = 1, N = Args.getNumElements(); i < N; ++i) {
    DIType Ty = resolve(Args.getElement(i));
    if (!Ty) {
      assert(i == N-1 && "Unspecified parameter must be the last argument");
      createAndAddDIE(dwarf::DW_TAG_unspecified_parameters, Buffer);
    } else {
      DIE &Arg = createAndAddDIE(dwarf::DW_TAG_formal_parameter, Buffer);
      addType(Arg, Ty);
      if (Ty.isArtificial())
        addFlag(Arg, dwarf::DW_AT_artificial);
    }
  }
}

/// constructTypeDIE - Construct type DIE from DICompositeType.
void DwarfUnit::constructTypeDIE(DIE &Buffer, DICompositeType CTy) {
  // Add name if not anonymous or intermediate type.
  StringRef Name = CTy.getName();

  uint64_t Size = CTy.getSizeInBits() >> 3;
  uint16_t Tag = Buffer.getTag();

  switch (Tag) {
  case dwarf::DW_TAG_array_type:
    constructArrayTypeDIE(Buffer, CTy);
    break;
  case dwarf::DW_TAG_enumeration_type:
    constructEnumTypeDIE(Buffer, CTy);
    break;
  case dwarf::DW_TAG_subroutine_type: {
    // Add return type. A void return won't have a type.
    DITypeArray Elements = DISubroutineType(CTy).getTypeArray();
    DIType RTy(resolve(Elements.getElement(0)));
    if (RTy)
      addType(Buffer, RTy);

    bool isPrototyped = true;
    if (Elements.getNumElements() == 2 &&
        !Elements.getElement(1))
      isPrototyped = false;

    constructSubprogramArguments(Buffer, Elements);

    // Add prototype flag if we're dealing with a C language and the
    // function has been prototyped.
    uint16_t Language = getLanguage();
    if (isPrototyped &&
        (Language == dwarf::DW_LANG_C89 || Language == dwarf::DW_LANG_C99 ||
         Language == dwarf::DW_LANG_ObjC))
      addFlag(Buffer, dwarf::DW_AT_prototyped);

    if (CTy.isLValueReference())
      addFlag(Buffer, dwarf::DW_AT_reference);

    if (CTy.isRValueReference())
      addFlag(Buffer, dwarf::DW_AT_rvalue_reference);
  } break;
  case dwarf::DW_TAG_structure_type:
  case dwarf::DW_TAG_union_type:
  case dwarf::DW_TAG_class_type: {
    // Add elements to structure type.
    DIArray Elements = CTy.getElements();
    for (unsigned i = 0, N = Elements.getNumElements(); i < N; ++i) {
      DIDescriptor Element = Elements.getElement(i);
      if (Element.isSubprogram())
        getOrCreateSubprogramDIE(DISubprogram(Element));
      else if (Element.isDerivedType()) {
        DIDerivedType DDTy(Element);
        if (DDTy.getTag() == dwarf::DW_TAG_friend) {
          DIE &ElemDie = createAndAddDIE(dwarf::DW_TAG_friend, Buffer);
          addType(ElemDie, resolve(DDTy.getTypeDerivedFrom()),
                  dwarf::DW_AT_friend);
        } else if (DDTy.isStaticMember()) {
          getOrCreateStaticMemberDIE(DDTy);
        } else {
          constructMemberDIE(Buffer, DDTy);
        }
      } else if (Element.isObjCProperty()) {
        DIObjCProperty Property(Element);
        DIE &ElemDie = createAndAddDIE(Property.getTag(), Buffer);
        StringRef PropertyName = Property.getObjCPropertyName();
        addString(ElemDie, dwarf::DW_AT_APPLE_property_name, PropertyName);
        if (Property.getType())
          addType(ElemDie, Property.getType());
        addSourceLine(ElemDie, Property);
        StringRef GetterName = Property.getObjCPropertyGetterName();
        if (!GetterName.empty())
          addString(ElemDie, dwarf::DW_AT_APPLE_property_getter, GetterName);
        StringRef SetterName = Property.getObjCPropertySetterName();
        if (!SetterName.empty())
          addString(ElemDie, dwarf::DW_AT_APPLE_property_setter, SetterName);
        unsigned PropertyAttributes = 0;
        if (Property.isReadOnlyObjCProperty())
          PropertyAttributes |= dwarf::DW_APPLE_PROPERTY_readonly;
        if (Property.isReadWriteObjCProperty())
          PropertyAttributes |= dwarf::DW_APPLE_PROPERTY_readwrite;
        if (Property.isAssignObjCProperty())
          PropertyAttributes |= dwarf::DW_APPLE_PROPERTY_assign;
        if (Property.isRetainObjCProperty())
          PropertyAttributes |= dwarf::DW_APPLE_PROPERTY_retain;
        if (Property.isCopyObjCProperty())
          PropertyAttributes |= dwarf::DW_APPLE_PROPERTY_copy;
        if (Property.isNonAtomicObjCProperty())
          PropertyAttributes |= dwarf::DW_APPLE_PROPERTY_nonatomic;
        if (PropertyAttributes)
          addUInt(ElemDie, dwarf::DW_AT_APPLE_property_attribute, None,
                  PropertyAttributes);

        DIEEntry *Entry = getDIEEntry(Element);
        if (!Entry) {
          Entry = createDIEEntry(ElemDie);
          insertDIEEntry(Element, Entry);
        }
      } else
        continue;
    }

    if (CTy.isAppleBlockExtension())
      addFlag(Buffer, dwarf::DW_AT_APPLE_block);

    // This is outside the DWARF spec, but GDB expects a DW_AT_containing_type
    // inside C++ composite types to point to the base class with the vtable.
    DICompositeType ContainingType(resolve(CTy.getContainingType()));
    if (ContainingType)
      addDIEEntry(Buffer, dwarf::DW_AT_containing_type,
                  *getOrCreateTypeDIE(ContainingType));

    if (CTy.isObjcClassComplete())
      addFlag(Buffer, dwarf::DW_AT_APPLE_objc_complete_type);

    // Add template parameters to a class, structure or union types.
    // FIXME: The support isn't in the metadata for this yet.
    if (Tag == dwarf::DW_TAG_class_type ||
        Tag == dwarf::DW_TAG_structure_type || Tag == dwarf::DW_TAG_union_type)
      addTemplateParams(Buffer, CTy.getTemplateParams());

    break;
  }
  default:
    break;
  }

  // Add name if not anonymous or intermediate type.
  if (!Name.empty())
    addString(Buffer, dwarf::DW_AT_name, Name);

  if (Tag == dwarf::DW_TAG_enumeration_type ||
      Tag == dwarf::DW_TAG_class_type || Tag == dwarf::DW_TAG_structure_type ||
      Tag == dwarf::DW_TAG_union_type) {
    // Add size if non-zero (derived types might be zero-sized.)
    // TODO: Do we care about size for enum forward declarations?
    if (Size)
      addUInt(Buffer, dwarf::DW_AT_byte_size, None, Size);
    else if (!CTy.isForwardDecl())
      // Add zero size if it is not a forward declaration.
      addUInt(Buffer, dwarf::DW_AT_byte_size, None, 0);

    // If we're a forward decl, say so.
    if (CTy.isForwardDecl())
      addFlag(Buffer, dwarf::DW_AT_declaration);

    // Add source line info if available.
    if (!CTy.isForwardDecl())
      addSourceLine(Buffer, CTy);

    // No harm in adding the runtime language to the declaration.
    unsigned RLang = CTy.getRunTimeLang();
    if (RLang)
      addUInt(Buffer, dwarf::DW_AT_APPLE_runtime_class, dwarf::DW_FORM_data1,
              RLang);
  }
}

/// constructTemplateTypeParameterDIE - Construct new DIE for the given
/// DITemplateTypeParameter.
void DwarfUnit::constructTemplateTypeParameterDIE(DIE &Buffer,
                                                  DITemplateTypeParameter TP) {
  DIE &ParamDIE =
      createAndAddDIE(dwarf::DW_TAG_template_type_parameter, Buffer);
  // Add the type if it exists, it could be void and therefore no type.
  if (TP.getType())
    addType(ParamDIE, resolve(TP.getType()));
  if (!TP.getName().empty())
    addString(ParamDIE, dwarf::DW_AT_name, TP.getName());
}

/// constructTemplateValueParameterDIE - Construct new DIE for the given
/// DITemplateValueParameter.
void
DwarfUnit::constructTemplateValueParameterDIE(DIE &Buffer,
                                              DITemplateValueParameter VP) {
  DIE &ParamDIE = createAndAddDIE(VP.getTag(), Buffer);

  // Add the type if there is one, template template and template parameter
  // packs will not have a type.
  if (VP.getTag() == dwarf::DW_TAG_template_value_parameter)
    addType(ParamDIE, resolve(VP.getType()));
  if (!VP.getName().empty())
    addString(ParamDIE, dwarf::DW_AT_name, VP.getName());
  if (Metadata *Val = VP.getValue()) {
    if (ConstantInt *CI = mdconst::dyn_extract<ConstantInt>(Val))
      addConstantValue(ParamDIE, CI, resolve(VP.getType()));
    else if (GlobalValue *GV = mdconst::dyn_extract<GlobalValue>(Val)) {
      // For declaration non-type template parameters (such as global values and
      // functions)
      DIELoc *Loc = new (DIEValueAllocator) DIELoc();
      addOpAddress(*Loc, Asm->getSymbol(GV));
      // Emit DW_OP_stack_value to use the address as the immediate value of the
      // parameter, rather than a pointer to it.
      addUInt(*Loc, dwarf::DW_FORM_data1, dwarf::DW_OP_stack_value);
      addBlock(ParamDIE, dwarf::DW_AT_location, Loc);
    } else if (VP.getTag() == dwarf::DW_TAG_GNU_template_template_param) {
      assert(isa<MDString>(Val));
      addString(ParamDIE, dwarf::DW_AT_GNU_template_name,
                cast<MDString>(Val)->getString());
    } else if (VP.getTag() == dwarf::DW_TAG_GNU_template_parameter_pack) {
      assert(isa<MDNode>(Val));
      DIArray A(cast<MDNode>(Val));
      addTemplateParams(ParamDIE, A);
    }
  }
}

/// getOrCreateNameSpace - Create a DIE for DINameSpace.
DIE *DwarfUnit::getOrCreateNameSpace(DINameSpace NS) {
  // Construct the context before querying for the existence of the DIE in case
  // such construction creates the DIE.
  DIE *ContextDIE = getOrCreateContextDIE(NS.getContext());

  if (DIE *NDie = getDIE(NS))
    return NDie;
  DIE &NDie = createAndAddDIE(dwarf::DW_TAG_namespace, *ContextDIE, NS);

  StringRef Name = NS.getName();
  if (!Name.empty())
    addString(NDie, dwarf::DW_AT_name, NS.getName());
  else
    Name = "(anonymous namespace)";
  DD->addAccelNamespace(Name, NDie);
  addGlobalName(Name, NDie, NS.getContext());
  addSourceLine(NDie, NS);
  return &NDie;
}

/// getOrCreateSubprogramDIE - Create new DIE using SP.
DIE *DwarfUnit::getOrCreateSubprogramDIE(DISubprogram SP, bool Minimal) {
  // Construct the context before querying for the existence of the DIE in case
  // such construction creates the DIE (as is the case for member function
  // declarations).
  DIE *ContextDIE =
      Minimal ? &getUnitDie() : getOrCreateContextDIE(resolve(SP.getContext()));

  if (DIE *SPDie = getDIE(SP))
    return SPDie;

  if (DISubprogram SPDecl = SP.getFunctionDeclaration()) {
    if (!Minimal) {
      // Add subprogram definitions to the CU die directly.
      ContextDIE = &getUnitDie();
      // Build the decl now to ensure it precedes the definition.
      getOrCreateSubprogramDIE(SPDecl);
    }
  }

  // DW_TAG_inlined_subroutine may refer to this DIE.
  DIE &SPDie = createAndAddDIE(dwarf::DW_TAG_subprogram, *ContextDIE, SP);

  // Stop here and fill this in later, depending on whether or not this
  // subprogram turns out to have inlined instances or not.
  if (SP.isDefinition())
    return &SPDie;

  applySubprogramAttributes(SP, SPDie);
  return &SPDie;
}

bool DwarfUnit::applySubprogramDefinitionAttributes(DISubprogram SP,
                                                    DIE &SPDie) {
  DIE *DeclDie = nullptr;
  StringRef DeclLinkageName;
  if (DISubprogram SPDecl = SP.getFunctionDeclaration()) {
    DeclDie = getDIE(SPDecl);
    assert(DeclDie && "This DIE should've already been constructed when the "
                      "definition DIE was created in "
                      "getOrCreateSubprogramDIE");
    DeclLinkageName = SPDecl.getLinkageName();
  }

  // Add function template parameters.
  addTemplateParams(SPDie, SP.getTemplateParams());

  // Add the linkage name if we have one and it isn't in the Decl.
  StringRef LinkageName = SP.getLinkageName();
  assert(((LinkageName.empty() || DeclLinkageName.empty()) ||
          LinkageName == DeclLinkageName) &&
         "decl has a linkage name and it is different");
  if (!LinkageName.empty() && DeclLinkageName.empty())
    addString(SPDie, dwarf::DW_AT_MIPS_linkage_name,
              GlobalValue::getRealLinkageName(LinkageName));

  if (!DeclDie)
    return false;

  // Refer to the function declaration where all the other attributes will be
  // found.
  addDIEEntry(SPDie, dwarf::DW_AT_specification, *DeclDie);
  return true;
}

void DwarfUnit::applySubprogramAttributes(DISubprogram SP, DIE &SPDie,
                                          bool Minimal) {
  if (!Minimal)
    if (applySubprogramDefinitionAttributes(SP, SPDie))
      return;

  // Constructors and operators for anonymous aggregates do not have names.
  if (!SP.getName().empty())
    addString(SPDie, dwarf::DW_AT_name, SP.getName());

  // Skip the rest of the attributes under -gmlt to save space.
  if (Minimal)
    return;

  addSourceLine(SPDie, SP);

  // Add the prototype if we have a prototype and we have a C like
  // language.
  uint16_t Language = getLanguage();
  if (SP.isPrototyped() &&
      (Language == dwarf::DW_LANG_C89 || Language == dwarf::DW_LANG_C99 ||
       Language == dwarf::DW_LANG_ObjC))
    addFlag(SPDie, dwarf::DW_AT_prototyped);

  DISubroutineType SPTy = SP.getType();
  assert(SPTy.getTag() == dwarf::DW_TAG_subroutine_type &&
         "the type of a subprogram should be a subroutine");

  DITypeArray Args = SPTy.getTypeArray();
  // Add a return type. If this is a type like a C/C++ void type we don't add a
  // return type.
  if (resolve(Args.getElement(0)))
    addType(SPDie, DIType(resolve(Args.getElement(0))));

  unsigned VK = SP.getVirtuality();
  if (VK) {
    addUInt(SPDie, dwarf::DW_AT_virtuality, dwarf::DW_FORM_data1, VK);
    DIELoc *Block = getDIELoc();
    addUInt(*Block, dwarf::DW_FORM_data1, dwarf::DW_OP_constu);
    addUInt(*Block, dwarf::DW_FORM_udata, SP.getVirtualIndex());
    addBlock(SPDie, dwarf::DW_AT_vtable_elem_location, Block);
    ContainingTypeMap.insert(
        std::make_pair(&SPDie, resolve(SP.getContainingType())));
  }

  if (!SP.isDefinition()) {
    addFlag(SPDie, dwarf::DW_AT_declaration);

    // Add arguments. Do not add arguments for subprogram definition. They will
    // be handled while processing variables.
    constructSubprogramArguments(SPDie, Args);
  }

  if (SP.isArtificial())
    addFlag(SPDie, dwarf::DW_AT_artificial);

  if (!SP.isLocalToUnit())
    addFlag(SPDie, dwarf::DW_AT_external);

  if (SP.isOptimized())
    addFlag(SPDie, dwarf::DW_AT_APPLE_optimized);

  if (unsigned isa = Asm->getISAEncoding()) {
    addUInt(SPDie, dwarf::DW_AT_APPLE_isa, dwarf::DW_FORM_flag, isa);
  }

  if (SP.isLValueReference())
    addFlag(SPDie, dwarf::DW_AT_reference);

  if (SP.isRValueReference())
    addFlag(SPDie, dwarf::DW_AT_rvalue_reference);

  if (SP.isProtected())
    addUInt(SPDie, dwarf::DW_AT_accessibility, dwarf::DW_FORM_data1,
            dwarf::DW_ACCESS_protected);
  else if (SP.isPrivate())
    addUInt(SPDie, dwarf::DW_AT_accessibility, dwarf::DW_FORM_data1,
            dwarf::DW_ACCESS_private);
  else if (SP.isPublic())
    addUInt(SPDie, dwarf::DW_AT_accessibility, dwarf::DW_FORM_data1,
            dwarf::DW_ACCESS_public);

  if (SP.isExplicit())
    addFlag(SPDie, dwarf::DW_AT_explicit);
}

/// constructSubrangeDIE - Construct subrange DIE from DISubrange.
void DwarfUnit::constructSubrangeDIE(DIE &Buffer, DISubrange SR, DIE *IndexTy) {
  DIE &DW_Subrange = createAndAddDIE(dwarf::DW_TAG_subrange_type, Buffer);
  addDIEEntry(DW_Subrange, dwarf::DW_AT_type, *IndexTy);

  // The LowerBound value defines the lower bounds which is typically zero for
  // C/C++. The Count value is the number of elements.  Values are 64 bit. If
  // Count == -1 then the array is unbounded and we do not emit
  // DW_AT_lower_bound and DW_AT_count attributes.
  int64_t LowerBound = SR.getLo();
  int64_t DefaultLowerBound = getDefaultLowerBound();
  int64_t Count = SR.getCount();

  if (DefaultLowerBound == -1 || LowerBound != DefaultLowerBound)
    addUInt(DW_Subrange, dwarf::DW_AT_lower_bound, None, LowerBound);

  if (Count != -1)
    // FIXME: An unbounded array should reference the expression that defines
    // the array.
    addUInt(DW_Subrange, dwarf::DW_AT_count, None, Count);
}

DIE *DwarfUnit::getIndexTyDie() {
  if (IndexTyDie)
    return IndexTyDie;
  // Construct an integer type to use for indexes.
  IndexTyDie = &createAndAddDIE(dwarf::DW_TAG_base_type, UnitDie);
  addString(*IndexTyDie, dwarf::DW_AT_name, "sizetype");
  addUInt(*IndexTyDie, dwarf::DW_AT_byte_size, None, sizeof(int64_t));
  addUInt(*IndexTyDie, dwarf::DW_AT_encoding, dwarf::DW_FORM_data1,
          dwarf::DW_ATE_unsigned);
  return IndexTyDie;
}

/// constructArrayTypeDIE - Construct array type DIE from DICompositeType.
void DwarfUnit::constructArrayTypeDIE(DIE &Buffer, DICompositeType CTy) {
  if (CTy.isVector())
    addFlag(Buffer, dwarf::DW_AT_GNU_vector);

  // Emit the element type.
  addType(Buffer, resolve(CTy.getTypeDerivedFrom()));

  // Get an anonymous type for index type.
  // FIXME: This type should be passed down from the front end
  // as different languages may have different sizes for indexes.
  DIE *IdxTy = getIndexTyDie();

  // Add subranges to array type.
  DIArray Elements = CTy.getElements();
  for (unsigned i = 0, N = Elements.getNumElements(); i < N; ++i) {
    DIDescriptor Element = Elements.getElement(i);
    if (Element.getTag() == dwarf::DW_TAG_subrange_type)
      constructSubrangeDIE(Buffer, DISubrange(Element), IdxTy);
  }
}

/// constructEnumTypeDIE - Construct an enum type DIE from DICompositeType.
void DwarfUnit::constructEnumTypeDIE(DIE &Buffer, DICompositeType CTy) {
  DIArray Elements = CTy.getElements();

  // Add enumerators to enumeration type.
  for (unsigned i = 0, N = Elements.getNumElements(); i < N; ++i) {
    DIEnumerator Enum(Elements.getElement(i));
    if (Enum.isEnumerator()) {
      DIE &Enumerator = createAndAddDIE(dwarf::DW_TAG_enumerator, Buffer);
      StringRef Name = Enum.getName();
      addString(Enumerator, dwarf::DW_AT_name, Name);
      int64_t Value = Enum.getEnumValue();
      addSInt(Enumerator, dwarf::DW_AT_const_value, dwarf::DW_FORM_sdata,
              Value);
    }
  }
  DIType DTy = resolve(CTy.getTypeDerivedFrom());
  if (DTy) {
    addType(Buffer, DTy);
    addFlag(Buffer, dwarf::DW_AT_enum_class);
  }
}

/// constructContainingTypeDIEs - Construct DIEs for types that contain
/// vtables.
void DwarfUnit::constructContainingTypeDIEs() {
  for (DenseMap<DIE *, const MDNode *>::iterator CI = ContainingTypeMap.begin(),
                                                 CE = ContainingTypeMap.end();
       CI != CE; ++CI) {
    DIE &SPDie = *CI->first;
    DIDescriptor D(CI->second);
    if (!D)
      continue;
    DIE *NDie = getDIE(D);
    if (!NDie)
      continue;
    addDIEEntry(SPDie, dwarf::DW_AT_containing_type, *NDie);
  }
}

/// constructMemberDIE - Construct member DIE from DIDerivedType.
void DwarfUnit::constructMemberDIE(DIE &Buffer, DIDerivedType DT) {
  DIE &MemberDie = createAndAddDIE(DT.getTag(), Buffer);
  StringRef Name = DT.getName();
  if (!Name.empty())
    addString(MemberDie, dwarf::DW_AT_name, Name);

  addType(MemberDie, resolve(DT.getTypeDerivedFrom()));

  addSourceLine(MemberDie, DT);

  if (DT.getTag() == dwarf::DW_TAG_inheritance && DT.isVirtual()) {

    // For C++, virtual base classes are not at fixed offset. Use following
    // expression to extract appropriate offset from vtable.
    // BaseAddr = ObAddr + *((*ObAddr) - Offset)

    DIELoc *VBaseLocationDie = new (DIEValueAllocator) DIELoc();
    addUInt(*VBaseLocationDie, dwarf::DW_FORM_data1, dwarf::DW_OP_dup);
    addUInt(*VBaseLocationDie, dwarf::DW_FORM_data1, dwarf::DW_OP_deref);
    addUInt(*VBaseLocationDie, dwarf::DW_FORM_data1, dwarf::DW_OP_constu);
    addUInt(*VBaseLocationDie, dwarf::DW_FORM_udata, DT.getOffsetInBits());
    addUInt(*VBaseLocationDie, dwarf::DW_FORM_data1, dwarf::DW_OP_minus);
    addUInt(*VBaseLocationDie, dwarf::DW_FORM_data1, dwarf::DW_OP_deref);
    addUInt(*VBaseLocationDie, dwarf::DW_FORM_data1, dwarf::DW_OP_plus);

    addBlock(MemberDie, dwarf::DW_AT_data_member_location, VBaseLocationDie);
  } else {
    uint64_t Size = DT.getSizeInBits();
    uint64_t FieldSize = getBaseTypeSize(DD, DT);
    uint64_t OffsetInBytes;

    if (FieldSize && Size != FieldSize) {
      // Handle bitfield, assume bytes are 8 bits.
      addUInt(MemberDie, dwarf::DW_AT_byte_size, None, FieldSize/8);
      addUInt(MemberDie, dwarf::DW_AT_bit_size, None, Size);

      uint64_t Offset = DT.getOffsetInBits();
      uint64_t AlignMask = ~(DT.getAlignInBits() - 1);
      uint64_t HiMark = (Offset + FieldSize) & AlignMask;
      uint64_t FieldOffset = (HiMark - FieldSize);
      Offset -= FieldOffset;

      // Maybe we need to work from the other end.
      if (Asm->getDataLayout().isLittleEndian())
        Offset = FieldSize - (Offset + Size);
      addUInt(MemberDie, dwarf::DW_AT_bit_offset, None, Offset);

      // Here DW_AT_data_member_location points to the anonymous
      // field that includes this bit field.
      OffsetInBytes = FieldOffset >> 3;
    } else
      // This is not a bitfield.
      OffsetInBytes = DT.getOffsetInBits() >> 3;

    if (DD->getDwarfVersion() <= 2) {
      DIELoc *MemLocationDie = new (DIEValueAllocator) DIELoc();
      addUInt(*MemLocationDie, dwarf::DW_FORM_data1, dwarf::DW_OP_plus_uconst);
      addUInt(*MemLocationDie, dwarf::DW_FORM_udata, OffsetInBytes);
      addBlock(MemberDie, dwarf::DW_AT_data_member_location, MemLocationDie);
    } else
      addUInt(MemberDie, dwarf::DW_AT_data_member_location, None,
              OffsetInBytes);
  }

  if (DT.isProtected())
    addUInt(MemberDie, dwarf::DW_AT_accessibility, dwarf::DW_FORM_data1,
            dwarf::DW_ACCESS_protected);
  else if (DT.isPrivate())
    addUInt(MemberDie, dwarf::DW_AT_accessibility, dwarf::DW_FORM_data1,
            dwarf::DW_ACCESS_private);
  // Otherwise C++ member and base classes are considered public.
  else if (DT.isPublic())
    addUInt(MemberDie, dwarf::DW_AT_accessibility, dwarf::DW_FORM_data1,
            dwarf::DW_ACCESS_public);
  if (DT.isVirtual())
    addUInt(MemberDie, dwarf::DW_AT_virtuality, dwarf::DW_FORM_data1,
            dwarf::DW_VIRTUALITY_virtual);

  // Objective-C properties.
  if (MDNode *PNode = DT.getObjCProperty())
    if (DIEEntry *PropertyDie = getDIEEntry(PNode))
      MemberDie.addValue(dwarf::DW_AT_APPLE_property, dwarf::DW_FORM_ref4,
                         PropertyDie);

  if (DT.isArtificial())
    addFlag(MemberDie, dwarf::DW_AT_artificial);
}

/// getOrCreateStaticMemberDIE - Create new DIE for C++ static member.
DIE *DwarfUnit::getOrCreateStaticMemberDIE(DIDerivedType DT) {
  if (!DT.Verify())
    return nullptr;

  // Construct the context before querying for the existence of the DIE in case
  // such construction creates the DIE.
  DIE *ContextDIE = getOrCreateContextDIE(resolve(DT.getContext()));
  assert(dwarf::isType(ContextDIE->getTag()) &&
         "Static member should belong to a type.");

  if (DIE *StaticMemberDIE = getDIE(DT))
    return StaticMemberDIE;

  DIE &StaticMemberDIE = createAndAddDIE(DT.getTag(), *ContextDIE, DT);

  DIType Ty = resolve(DT.getTypeDerivedFrom());

  addString(StaticMemberDIE, dwarf::DW_AT_name, DT.getName());
  addType(StaticMemberDIE, Ty);
  addSourceLine(StaticMemberDIE, DT);
  addFlag(StaticMemberDIE, dwarf::DW_AT_external);
  addFlag(StaticMemberDIE, dwarf::DW_AT_declaration);

  // FIXME: We could omit private if the parent is a class_type, and
  // public if the parent is something else.
  if (DT.isProtected())
    addUInt(StaticMemberDIE, dwarf::DW_AT_accessibility, dwarf::DW_FORM_data1,
            dwarf::DW_ACCESS_protected);
  else if (DT.isPrivate())
    addUInt(StaticMemberDIE, dwarf::DW_AT_accessibility, dwarf::DW_FORM_data1,
            dwarf::DW_ACCESS_private);
  else if (DT.isPublic())
    addUInt(StaticMemberDIE, dwarf::DW_AT_accessibility, dwarf::DW_FORM_data1,
            dwarf::DW_ACCESS_public);

  if (const ConstantInt *CI = dyn_cast_or_null<ConstantInt>(DT.getConstant()))
    addConstantValue(StaticMemberDIE, CI, Ty);
  if (const ConstantFP *CFP = dyn_cast_or_null<ConstantFP>(DT.getConstant()))
    addConstantFPValue(StaticMemberDIE, CFP);

  return &StaticMemberDIE;
}

void DwarfUnit::emitHeader(const MCSymbol *ASectionSym) const {
  // Emit size of content not including length itself
  Asm->OutStreamer.AddComment("Length of Unit");
  Asm->EmitInt32(getHeaderSize() + UnitDie.getSize());

  Asm->OutStreamer.AddComment("DWARF version number");
  Asm->EmitInt16(DD->getDwarfVersion());
  Asm->OutStreamer.AddComment("Offset Into Abbrev. Section");
  // We share one abbreviations table across all units so it's always at the
  // start of the section. Use a relocatable offset where needed to ensure
  // linking doesn't invalidate that offset.
  if (ASectionSym)
    Asm->EmitSectionOffset(ASectionSym, ASectionSym);
  else
    // Use a constant value when no symbol is provided.
    Asm->EmitInt32(0);
  Asm->OutStreamer.AddComment("Address Size (in bytes)");
  Asm->EmitInt8(Asm->getDataLayout().getPointerSize());
}

void DwarfUnit::initSection(const MCSection *Section) {
  assert(!this->Section);
  this->Section = Section;
}

void DwarfTypeUnit::emitHeader(const MCSymbol *ASectionSym) const {
  DwarfUnit::emitHeader(ASectionSym);
  Asm->OutStreamer.AddComment("Type Signature");
  Asm->OutStreamer.EmitIntValue(TypeSignature, sizeof(TypeSignature));
  Asm->OutStreamer.AddComment("Type DIE Offset");
  // In a skeleton type unit there is no type DIE so emit a zero offset.
  Asm->OutStreamer.EmitIntValue(Ty ? Ty->getOffset() : 0,
                                sizeof(Ty->getOffset()));
}

bool DwarfTypeUnit::isDwoUnit() const {
  // Since there are no skeleton type units, all type units are dwo type units
  // when split DWARF is being used.
  return DD->useSplitDwarf();
}
