//===-- llvm/CodeGen/DwarfCompileUnit.cpp - Dwarf Compile Unit ------------===//
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

#define DEBUG_TYPE "dwarfdebug"

#include "DwarfCompileUnit.h"
#include "DwarfDebug.h"
#include "llvm/Constants.h"
#include "llvm/GlobalVariable.h"
#include "llvm/Instructions.h"
#include "llvm/Analysis/DIBuilder.h"
#include "llvm/Target/Mangler.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetFrameLowering.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

/// CompileUnit - Compile unit constructor.
CompileUnit::CompileUnit(unsigned I, DIE *D, AsmPrinter *A, DwarfDebug *DW)
  : ID(I), CUDie(D), Asm(A), DD(DW), IndexTyDie(0) {
  DIEIntegerOne = new (DIEValueAllocator) DIEInteger(1);
}

/// ~CompileUnit - Destructor for compile unit.
CompileUnit::~CompileUnit() {
  for (unsigned j = 0, M = DIEBlocks.size(); j < M; ++j)
    DIEBlocks[j]->~DIEBlock();
}

/// createDIEEntry - Creates a new DIEEntry to be a proxy for a debug
/// information entry.
DIEEntry *CompileUnit::createDIEEntry(DIE *Entry) {
  DIEEntry *Value = new (DIEValueAllocator) DIEEntry(Entry);
  return Value;
}

/// addUInt - Add an unsigned integer attribute data and value.
///
void CompileUnit::addUInt(DIE *Die, unsigned Attribute,
                          unsigned Form, uint64_t Integer) {
  if (!Form) Form = DIEInteger::BestForm(false, Integer);
  DIEValue *Value = Integer == 1 ?
    DIEIntegerOne : new (DIEValueAllocator) DIEInteger(Integer);
  Die->addValue(Attribute, Form, Value);
}

/// addSInt - Add an signed integer attribute data and value.
///
void CompileUnit::addSInt(DIE *Die, unsigned Attribute,
                          unsigned Form, int64_t Integer) {
  if (!Form) Form = DIEInteger::BestForm(true, Integer);
  DIEValue *Value = new (DIEValueAllocator) DIEInteger(Integer);
  Die->addValue(Attribute, Form, Value);
}

/// addString - Add a string attribute data and value. DIEString only
/// keeps string reference.
void CompileUnit::addString(DIE *Die, unsigned Attribute, unsigned Form,
                            StringRef String) {
  DIEValue *Value = new (DIEValueAllocator) DIEString(String);
  Die->addValue(Attribute, Form, Value);
}

/// addLabel - Add a Dwarf label attribute data and value.
///
void CompileUnit::addLabel(DIE *Die, unsigned Attribute, unsigned Form,
                           const MCSymbol *Label) {
  DIEValue *Value = new (DIEValueAllocator) DIELabel(Label);
  Die->addValue(Attribute, Form, Value);
}

/// addDelta - Add a label delta attribute data and value.
///
void CompileUnit::addDelta(DIE *Die, unsigned Attribute, unsigned Form,
                           const MCSymbol *Hi, const MCSymbol *Lo) {
  DIEValue *Value = new (DIEValueAllocator) DIEDelta(Hi, Lo);
  Die->addValue(Attribute, Form, Value);
}

/// addDIEEntry - Add a DIE attribute data and value.
///
void CompileUnit::addDIEEntry(DIE *Die, unsigned Attribute, unsigned Form,
                              DIE *Entry) {
  Die->addValue(Attribute, Form, createDIEEntry(Entry));
}


/// addBlock - Add block data.
///
void CompileUnit::addBlock(DIE *Die, unsigned Attribute, unsigned Form,
                           DIEBlock *Block) {
  Block->ComputeSize(Asm);
  DIEBlocks.push_back(Block); // Memoize so we can call the destructor later on.
  Die->addValue(Attribute, Block->BestForm(), Block);
}

/// addSourceLine - Add location information to specified debug information
/// entry.
void CompileUnit::addSourceLine(DIE *Die, DIVariable V) {
  // Verify variable.
  if (!V.Verify())
    return;
  
  unsigned Line = V.getLineNumber();
  if (Line == 0)
    return;
  unsigned FileID = DD->GetOrCreateSourceID(V.getContext().getFilename(),
                                            V.getContext().getDirectory());
  assert(FileID && "Invalid file id");
  addUInt(Die, dwarf::DW_AT_decl_file, 0, FileID);
  addUInt(Die, dwarf::DW_AT_decl_line, 0, Line);
}

/// addSourceLine - Add location information to specified debug information
/// entry.
void CompileUnit::addSourceLine(DIE *Die, DIGlobalVariable G) {
  // Verify global variable.
  if (!G.Verify())
    return;

  unsigned Line = G.getLineNumber();
  if (Line == 0)
    return;
  unsigned FileID = DD->GetOrCreateSourceID(G.getFilename(),
                                            G.getDirectory());
  assert(FileID && "Invalid file id");
  addUInt(Die, dwarf::DW_AT_decl_file, 0, FileID);
  addUInt(Die, dwarf::DW_AT_decl_line, 0, Line);
}

/// addSourceLine - Add location information to specified debug information
/// entry.
void CompileUnit::addSourceLine(DIE *Die, DISubprogram SP) {
  // Verify subprogram.
  if (!SP.Verify())
    return;
  // If the line number is 0, don't add it.
  if (SP.getLineNumber() == 0)
    return;

  unsigned Line = SP.getLineNumber();
  if (!SP.getContext().Verify())
    return;
  unsigned FileID = DD->GetOrCreateSourceID(SP.getFilename(), SP.getDirectory());
  assert(FileID && "Invalid file id");
  addUInt(Die, dwarf::DW_AT_decl_file, 0, FileID);
  addUInt(Die, dwarf::DW_AT_decl_line, 0, Line);
}

/// addSourceLine - Add location information to specified debug information
/// entry.
void CompileUnit::addSourceLine(DIE *Die, DIType Ty) {
  // Verify type.
  if (!Ty.Verify())
    return;

  unsigned Line = Ty.getLineNumber();
  if (Line == 0 || !Ty.getContext().Verify())
    return;
  unsigned FileID = DD->GetOrCreateSourceID(Ty.getFilename(), Ty.getDirectory());
  assert(FileID && "Invalid file id");
  addUInt(Die, dwarf::DW_AT_decl_file, 0, FileID);
  addUInt(Die, dwarf::DW_AT_decl_line, 0, Line);
}

/// addSourceLine - Add location information to specified debug information
/// entry.
void CompileUnit::addSourceLine(DIE *Die, DINameSpace NS) {
  // Verify namespace.
  if (!NS.Verify())
    return;

  unsigned Line = NS.getLineNumber();
  if (Line == 0)
    return;
  StringRef FN = NS.getFilename();

  unsigned FileID = DD->GetOrCreateSourceID(FN, NS.getDirectory());
  assert(FileID && "Invalid file id");
  addUInt(Die, dwarf::DW_AT_decl_file, 0, FileID);
  addUInt(Die, dwarf::DW_AT_decl_line, 0, Line);
}

/// addVariableAddress - Add DW_AT_location attribute for a 
/// DbgVariable based on provided MachineLocation.
void CompileUnit::addVariableAddress(DbgVariable *&DV, DIE *Die, 
                                     MachineLocation Location) {
  if (DV->variableHasComplexAddress())
    addComplexAddress(DV, Die, dwarf::DW_AT_location, Location);
  else if (DV->isBlockByrefVariable())
    addBlockByrefAddress(DV, Die, dwarf::DW_AT_location, Location);
  else
    addAddress(Die, dwarf::DW_AT_location, Location);
}

/// addRegisterOp - Add register operand.
void CompileUnit::addRegisterOp(DIE *TheDie, unsigned Reg) {
  const TargetRegisterInfo *RI = Asm->TM.getRegisterInfo();
  unsigned DWReg = RI->getDwarfRegNum(Reg, false);
  if (DWReg < 32)
    addUInt(TheDie, 0, dwarf::DW_FORM_data1, dwarf::DW_OP_reg0 + DWReg);
  else {
    addUInt(TheDie, 0, dwarf::DW_FORM_data1, dwarf::DW_OP_regx);
    addUInt(TheDie, 0, dwarf::DW_FORM_udata, DWReg);
  }
}

/// addRegisterOffset - Add register offset.
void CompileUnit::addRegisterOffset(DIE *TheDie, unsigned Reg,
                                    int64_t Offset) {
  const TargetRegisterInfo *RI = Asm->TM.getRegisterInfo();
  unsigned DWReg = RI->getDwarfRegNum(Reg, false);
  const TargetRegisterInfo *TRI = Asm->TM.getRegisterInfo();
  if (Reg == TRI->getFrameRegister(*Asm->MF))
    // If variable offset is based in frame register then use fbreg.
    addUInt(TheDie, 0, dwarf::DW_FORM_data1, dwarf::DW_OP_fbreg);
  else if (DWReg < 32)
    addUInt(TheDie, 0, dwarf::DW_FORM_data1, dwarf::DW_OP_breg0 + DWReg);
  else {
    addUInt(TheDie, 0, dwarf::DW_FORM_data1, dwarf::DW_OP_bregx);
    addUInt(TheDie, 0, dwarf::DW_FORM_udata, DWReg);
  }
  addSInt(TheDie, 0, dwarf::DW_FORM_sdata, Offset);
}

/// addAddress - Add an address attribute to a die based on the location
/// provided.
void CompileUnit::addAddress(DIE *Die, unsigned Attribute,
                             const MachineLocation &Location) {
  DIEBlock *Block = new (DIEValueAllocator) DIEBlock();

  if (Location.isReg())
    addRegisterOp(Block, Location.getReg());
  else
    addRegisterOffset(Block, Location.getReg(), Location.getOffset());

  // Now attach the location information to the DIE.
  addBlock(Die, Attribute, 0, Block);
}

/// addComplexAddress - Start with the address based on the location provided,
/// and generate the DWARF information necessary to find the actual variable
/// given the extra address information encoded in the DIVariable, starting from
/// the starting location.  Add the DWARF information to the die.
///
void CompileUnit::addComplexAddress(DbgVariable *&DV, DIE *Die,
                                    unsigned Attribute,
                                    const MachineLocation &Location) {
  DIEBlock *Block = new (DIEValueAllocator) DIEBlock();
  unsigned N = DV->getNumAddrElements();
  unsigned i = 0;
  if (Location.isReg()) {
    if (N >= 2 && DV->getAddrElement(0) == DIBuilder::OpPlus) {
      // If first address element is OpPlus then emit
      // DW_OP_breg + Offset instead of DW_OP_reg + Offset.
      addRegisterOffset(Block, Location.getReg(), DV->getAddrElement(1));
      i = 2;
    } else
      addRegisterOp(Block, Location.getReg());
  }
  else
    addRegisterOffset(Block, Location.getReg(), Location.getOffset());

  for (;i < N; ++i) {
    uint64_t Element = DV->getAddrElement(i);
    if (Element == DIBuilder::OpPlus) {
      addUInt(Block, 0, dwarf::DW_FORM_data1, dwarf::DW_OP_plus_uconst);
      addUInt(Block, 0, dwarf::DW_FORM_udata, DV->getAddrElement(++i));
    } else if (Element == DIBuilder::OpDeref) {
      addUInt(Block, 0, dwarf::DW_FORM_data1, dwarf::DW_OP_deref);
    } else llvm_unreachable("unknown DIBuilder Opcode");
  }

  // Now attach the location information to the DIE.
  addBlock(Die, Attribute, 0, Block);
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
void CompileUnit::addBlockByrefAddress(DbgVariable *&DV, DIE *Die,
                                       unsigned Attribute,
                                       const MachineLocation &Location) {
  DIType Ty = DV->getType();
  DIType TmpTy = Ty;
  unsigned Tag = Ty.getTag();
  bool isPointer = false;

  StringRef varName = DV->getName();

  if (Tag == dwarf::DW_TAG_pointer_type) {
    DIDerivedType DTy = DIDerivedType(Ty);
    TmpTy = DTy.getTypeDerivedFrom();
    isPointer = true;
  }

  DICompositeType blockStruct = DICompositeType(TmpTy);

  // Find the __forwarding field and the variable field in the __Block_byref
  // struct.
  DIArray Fields = blockStruct.getTypeArray();
  DIDescriptor varField = DIDescriptor();
  DIDescriptor forwardingField = DIDescriptor();

  for (unsigned i = 0, N = Fields.getNumElements(); i < N; ++i) {
    DIDescriptor Element = Fields.getElement(i);
    DIDerivedType DT = DIDerivedType(Element);
    StringRef fieldName = DT.getName();
    if (fieldName == "__forwarding")
      forwardingField = Element;
    else if (fieldName == varName)
      varField = Element;
  }

  // Get the offsets for the forwarding field and the variable field.
  unsigned forwardingFieldOffset =
    DIDerivedType(forwardingField).getOffsetInBits() >> 3;
  unsigned varFieldOffset =
    DIDerivedType(varField).getOffsetInBits() >> 3;

  // Decode the original location, and use that as the start of the byref
  // variable's location.
  const TargetRegisterInfo *RI = Asm->TM.getRegisterInfo();
  unsigned Reg = RI->getDwarfRegNum(Location.getReg(), false);
  DIEBlock *Block = new (DIEValueAllocator) DIEBlock();

  if (Location.isReg()) {
    if (Reg < 32)
      addUInt(Block, 0, dwarf::DW_FORM_data1, dwarf::DW_OP_reg0 + Reg);
    else {
      addUInt(Block, 0, dwarf::DW_FORM_data1, dwarf::DW_OP_regx);
      addUInt(Block, 0, dwarf::DW_FORM_udata, Reg);
    }
  } else {
    if (Reg < 32)
      addUInt(Block, 0, dwarf::DW_FORM_data1, dwarf::DW_OP_breg0 + Reg);
    else {
      addUInt(Block, 0, dwarf::DW_FORM_data1, dwarf::DW_OP_bregx);
      addUInt(Block, 0, dwarf::DW_FORM_udata, Reg);
    }

    addUInt(Block, 0, dwarf::DW_FORM_sdata, Location.getOffset());
  }

  // If we started with a pointer to the __Block_byref... struct, then
  // the first thing we need to do is dereference the pointer (DW_OP_deref).
  if (isPointer)
    addUInt(Block, 0, dwarf::DW_FORM_data1, dwarf::DW_OP_deref);

  // Next add the offset for the '__forwarding' field:
  // DW_OP_plus_uconst ForwardingFieldOffset.  Note there's no point in
  // adding the offset if it's 0.
  if (forwardingFieldOffset > 0) {
    addUInt(Block, 0, dwarf::DW_FORM_data1, dwarf::DW_OP_plus_uconst);
    addUInt(Block, 0, dwarf::DW_FORM_udata, forwardingFieldOffset);
  }

  // Now dereference the __forwarding field to get to the real __Block_byref
  // struct:  DW_OP_deref.
  addUInt(Block, 0, dwarf::DW_FORM_data1, dwarf::DW_OP_deref);

  // Now that we've got the real __Block_byref... struct, add the offset
  // for the variable's field to get to the location of the actual variable:
  // DW_OP_plus_uconst varFieldOffset.  Again, don't add if it's 0.
  if (varFieldOffset > 0) {
    addUInt(Block, 0, dwarf::DW_FORM_data1, dwarf::DW_OP_plus_uconst);
    addUInt(Block, 0, dwarf::DW_FORM_udata, varFieldOffset);
  }

  // Now attach the location information to the DIE.
  addBlock(Die, Attribute, 0, Block);
}

/// isTypeSigned - Return true if the type is signed.
static bool isTypeSigned(DIType Ty, int *SizeInBits) {
  if (Ty.isDerivedType())
    return isTypeSigned(DIDerivedType(Ty).getTypeDerivedFrom(), SizeInBits);
  if (Ty.isBasicType())
    if (DIBasicType(Ty).getEncoding() == dwarf::DW_ATE_signed
        || DIBasicType(Ty).getEncoding() == dwarf::DW_ATE_signed_char) {
      *SizeInBits = Ty.getSizeInBits();
      return true;
    }
  return false;
}

/// addConstantValue - Add constant value entry in variable DIE.
bool CompileUnit::addConstantValue(DIE *Die, const MachineOperand &MO,
                                   DIType Ty) {
  assert (MO.isImm() && "Invalid machine operand!");
  DIEBlock *Block = new (DIEValueAllocator) DIEBlock();
  int SizeInBits = -1;
  bool SignedConstant = isTypeSigned(Ty, &SizeInBits);
  unsigned Form = SignedConstant ? dwarf::DW_FORM_sdata : dwarf::DW_FORM_udata;
  switch (SizeInBits) {
    case 8:  Form = dwarf::DW_FORM_data1; break;
    case 16: Form = dwarf::DW_FORM_data2; break;
    case 32: Form = dwarf::DW_FORM_data4; break;
    case 64: Form = dwarf::DW_FORM_data8; break;
    default: break;
  }
  SignedConstant ? addSInt(Block, 0, Form, MO.getImm()) 
    : addUInt(Block, 0, Form, MO.getImm());

  addBlock(Die, dwarf::DW_AT_const_value, 0, Block);
  return true;
}

/// addConstantFPValue - Add constant value entry in variable DIE.
bool CompileUnit::addConstantFPValue(DIE *Die, const MachineOperand &MO) {
  assert (MO.isFPImm() && "Invalid machine operand!");
  DIEBlock *Block = new (DIEValueAllocator) DIEBlock();
  APFloat FPImm = MO.getFPImm()->getValueAPF();

  // Get the raw data form of the floating point.
  const APInt FltVal = FPImm.bitcastToAPInt();
  const char *FltPtr = (const char*)FltVal.getRawData();

  int NumBytes = FltVal.getBitWidth() / 8; // 8 bits per byte.
  bool LittleEndian = Asm->getTargetData().isLittleEndian();
  int Incr = (LittleEndian ? 1 : -1);
  int Start = (LittleEndian ? 0 : NumBytes - 1);
  int Stop = (LittleEndian ? NumBytes : -1);

  // Output the constant to DWARF one byte at a time.
  for (; Start != Stop; Start += Incr)
    addUInt(Block, 0, dwarf::DW_FORM_data1,
            (unsigned char)0xFF & FltPtr[Start]);

  addBlock(Die, dwarf::DW_AT_const_value, 0, Block);
  return true;
}

/// addConstantValue - Add constant value entry in variable DIE.
bool CompileUnit::addConstantValue(DIE *Die, const ConstantInt *CI,
                                   bool Unsigned) {
  unsigned CIBitWidth = CI->getBitWidth();
  if (CIBitWidth <= 64) {
    unsigned form = 0;
    switch (CIBitWidth) {
    case 8: form = dwarf::DW_FORM_data1; break;
    case 16: form = dwarf::DW_FORM_data2; break;
    case 32: form = dwarf::DW_FORM_data4; break;
    case 64: form = dwarf::DW_FORM_data8; break;
    default: 
      form = Unsigned ? dwarf::DW_FORM_udata : dwarf::DW_FORM_sdata;
    }
    if (Unsigned)
      addUInt(Die, dwarf::DW_AT_const_value, form, CI->getZExtValue());
    else
      addSInt(Die, dwarf::DW_AT_const_value, form, CI->getSExtValue());
    return true;
  }

  DIEBlock *Block = new (DIEValueAllocator) DIEBlock();

  // Get the raw data form of the large APInt.
  const APInt Val = CI->getValue();
  const uint64_t *Ptr64 = Val.getRawData();

  int NumBytes = Val.getBitWidth() / 8; // 8 bits per byte.
  bool LittleEndian = Asm->getTargetData().isLittleEndian();

  // Output the constant to DWARF one byte at a time.
  for (int i = 0; i < NumBytes; i++) {
    uint8_t c;
    if (LittleEndian)
      c = Ptr64[i / 8] >> (8 * (i & 7));
    else
      c = Ptr64[(NumBytes - 1 - i) / 8] >> (8 * ((NumBytes - 1 - i) & 7));
    addUInt(Block, 0, dwarf::DW_FORM_data1, c);
  }

  addBlock(Die, dwarf::DW_AT_const_value, 0, Block);
  return true;
}

/// addTemplateParams - Add template parameters in buffer.
void CompileUnit::addTemplateParams(DIE &Buffer, DIArray TParams) {
  // Add template parameters.
  for (unsigned i = 0, e = TParams.getNumElements(); i != e; ++i) {
    DIDescriptor Element = TParams.getElement(i);
    if (Element.isTemplateTypeParameter())
      Buffer.addChild(getOrCreateTemplateTypeParameterDIE(
                        DITemplateTypeParameter(Element)));
    else if (Element.isTemplateValueParameter())
      Buffer.addChild(getOrCreateTemplateValueParameterDIE(
                        DITemplateValueParameter(Element)));
  }

}
/// addToContextOwner - Add Die into the list of its context owner's children.
void CompileUnit::addToContextOwner(DIE *Die, DIDescriptor Context) {
  if (Context.isType()) {
    DIE *ContextDIE = getOrCreateTypeDIE(DIType(Context));
    ContextDIE->addChild(Die);
  } else if (Context.isNameSpace()) {
    DIE *ContextDIE = getOrCreateNameSpace(DINameSpace(Context));
    ContextDIE->addChild(Die);
  } else if (Context.isSubprogram()) {
    DIE *ContextDIE = getOrCreateSubprogramDIE(DISubprogram(Context));
    ContextDIE->addChild(Die);
  } else if (DIE *ContextDIE = getDIE(Context))
    ContextDIE->addChild(Die);
  else
    addDie(Die);
}

/// getOrCreateTypeDIE - Find existing DIE or create new DIE for the
/// given DIType.
DIE *CompileUnit::getOrCreateTypeDIE(const MDNode *TyNode) {
  DIType Ty(TyNode);
  if (!Ty.Verify())
    return NULL;
  DIE *TyDIE = getDIE(Ty);
  if (TyDIE)
    return TyDIE;

  // Create new type.
  TyDIE = new DIE(dwarf::DW_TAG_base_type);
  insertDIE(Ty, TyDIE);
  if (Ty.isBasicType())
    constructTypeDIE(*TyDIE, DIBasicType(Ty));
  else if (Ty.isCompositeType())
    constructTypeDIE(*TyDIE, DICompositeType(Ty));
  else {
    assert(Ty.isDerivedType() && "Unknown kind of DIType");
    constructTypeDIE(*TyDIE, DIDerivedType(Ty));
  }

  addToContextOwner(TyDIE, Ty.getContext());
  return TyDIE;
}

/// addType - Add a new type attribute to the specified entity.
void CompileUnit::addType(DIE *Entity, DIType Ty) {
  if (!Ty.Verify())
    return;

  // Check for pre-existence.
  DIEEntry *Entry = getDIEEntry(Ty);
  // If it exists then use the existing value.
  if (Entry) {
    Entity->addValue(dwarf::DW_AT_type, dwarf::DW_FORM_ref4, Entry);
    return;
  }

  // Construct type.
  DIE *Buffer = getOrCreateTypeDIE(Ty);

  // Set up proxy.
  Entry = createDIEEntry(Buffer);
  insertDIEEntry(Ty, Entry);
  Entity->addValue(dwarf::DW_AT_type, dwarf::DW_FORM_ref4, Entry);

  // If this is a complete composite type then include it in the
  // list of global types.
  addGlobalType(Ty);
}

/// addGlobalType - Add a new global type to the compile unit.
///
void CompileUnit::addGlobalType(DIType Ty) {
  DIDescriptor Context = Ty.getContext();
  if (Ty.isCompositeType() && !Ty.getName().empty() && !Ty.isForwardDecl() 
      && (!Context || Context.isCompileUnit() || Context.isFile() 
          || Context.isNameSpace()))
    if (DIEEntry *Entry = getDIEEntry(Ty))
      GlobalTypes[Ty.getName()] = Entry->getEntry();
}

/// addPubTypes - Add type for pubtypes section.
void CompileUnit::addPubTypes(DISubprogram SP) {
  DICompositeType SPTy = SP.getType();
  unsigned SPTag = SPTy.getTag();
  if (SPTag != dwarf::DW_TAG_subroutine_type)
    return;

  DIArray Args = SPTy.getTypeArray();
  for (unsigned i = 0, e = Args.getNumElements(); i != e; ++i) {
    DIType ATy(Args.getElement(i));
    if (!ATy.Verify())
      continue;
    addGlobalType(ATy);
  }
}

/// constructTypeDIE - Construct basic type die from DIBasicType.
void CompileUnit::constructTypeDIE(DIE &Buffer, DIBasicType BTy) {
  // Get core information.
  StringRef Name = BTy.getName();
  // Add name if not anonymous or intermediate type.
  if (!Name.empty())
    addString(&Buffer, dwarf::DW_AT_name, dwarf::DW_FORM_string, Name);

  if (BTy.getTag() == dwarf::DW_TAG_unspecified_type) {
    Buffer.setTag(dwarf::DW_TAG_unspecified_type);
    // Unspecified types has only name, nothing else.
    return;
  }

  Buffer.setTag(dwarf::DW_TAG_base_type);
  addUInt(&Buffer, dwarf::DW_AT_encoding,  dwarf::DW_FORM_data1,
	  BTy.getEncoding());

  uint64_t Size = BTy.getSizeInBits() >> 3;
  addUInt(&Buffer, dwarf::DW_AT_byte_size, 0, Size);
}

/// constructTypeDIE - Construct derived type die from DIDerivedType.
void CompileUnit::constructTypeDIE(DIE &Buffer, DIDerivedType DTy) {
  // Get core information.
  StringRef Name = DTy.getName();
  uint64_t Size = DTy.getSizeInBits() >> 3;
  unsigned Tag = DTy.getTag();

  // FIXME - Workaround for templates.
  if (Tag == dwarf::DW_TAG_inheritance) Tag = dwarf::DW_TAG_reference_type;

  Buffer.setTag(Tag);

  // Map to main type, void will not have a type.
  DIType FromTy = DTy.getTypeDerivedFrom();
  addType(&Buffer, FromTy);

  // Add name if not anonymous or intermediate type.
  if (!Name.empty())
    addString(&Buffer, dwarf::DW_AT_name, dwarf::DW_FORM_string, Name);

  // Add size if non-zero (derived types might be zero-sized.)
  if (Size)
    addUInt(&Buffer, dwarf::DW_AT_byte_size, 0, Size);

  // Add source line info if available and TyDesc is not a forward declaration.
  if (!DTy.isForwardDecl())
    addSourceLine(&Buffer, DTy);
}

/// constructTypeDIE - Construct type DIE from DICompositeType.
void CompileUnit::constructTypeDIE(DIE &Buffer, DICompositeType CTy) {
  // Get core information.
  StringRef Name = CTy.getName();

  uint64_t Size = CTy.getSizeInBits() >> 3;
  unsigned Tag = CTy.getTag();
  Buffer.setTag(Tag);

  switch (Tag) {
  case dwarf::DW_TAG_vector_type:
  case dwarf::DW_TAG_array_type:
    constructArrayTypeDIE(Buffer, &CTy);
    break;
  case dwarf::DW_TAG_enumeration_type: {
    DIArray Elements = CTy.getTypeArray();

    // Add enumerators to enumeration type.
    for (unsigned i = 0, N = Elements.getNumElements(); i < N; ++i) {
      DIE *ElemDie = NULL;
      DIDescriptor Enum(Elements.getElement(i));
      if (Enum.isEnumerator()) {
        ElemDie = constructEnumTypeDIE(DIEnumerator(Enum));
        Buffer.addChild(ElemDie);
      }
    }
  }
    break;
  case dwarf::DW_TAG_subroutine_type: {
    // Add return type.
    DIArray Elements = CTy.getTypeArray();
    DIDescriptor RTy = Elements.getElement(0);
    addType(&Buffer, DIType(RTy));

    bool isPrototyped = true;
    // Add arguments.
    for (unsigned i = 1, N = Elements.getNumElements(); i < N; ++i) {
      DIDescriptor Ty = Elements.getElement(i);
      if (Ty.isUnspecifiedParameter()) {
        DIE *Arg = new DIE(dwarf::DW_TAG_unspecified_parameters);
        Buffer.addChild(Arg);
        isPrototyped = false;
      } else {
        DIE *Arg = new DIE(dwarf::DW_TAG_formal_parameter);
        addType(Arg, DIType(Ty));
        Buffer.addChild(Arg);
      }
    }
    // Add prototype flag.
    if (isPrototyped)
      addUInt(&Buffer, dwarf::DW_AT_prototyped, dwarf::DW_FORM_flag, 1);
  }
    break;
  case dwarf::DW_TAG_structure_type:
  case dwarf::DW_TAG_union_type:
  case dwarf::DW_TAG_class_type: {
    // Add elements to structure type.
    DIArray Elements = CTy.getTypeArray();

    // A forward struct declared type may not have elements available.
    unsigned N = Elements.getNumElements();
    if (N == 0)
      break;

    // Add elements to structure type.
    for (unsigned i = 0; i < N; ++i) {
      DIDescriptor Element = Elements.getElement(i);
      DIE *ElemDie = NULL;
      if (Element.isSubprogram()) {
        DISubprogram SP(Element);
        ElemDie = getOrCreateSubprogramDIE(DISubprogram(Element));
        if (SP.isProtected())
          addUInt(ElemDie, dwarf::DW_AT_accessibility, dwarf::DW_FORM_flag,
                  dwarf::DW_ACCESS_protected);
        else if (SP.isPrivate())
          addUInt(ElemDie, dwarf::DW_AT_accessibility, dwarf::DW_FORM_flag,
                  dwarf::DW_ACCESS_private);
        else 
          addUInt(ElemDie, dwarf::DW_AT_accessibility, dwarf::DW_FORM_flag,
            dwarf::DW_ACCESS_public);
        if (SP.isExplicit())
          addUInt(ElemDie, dwarf::DW_AT_explicit, dwarf::DW_FORM_flag, 1);
      }
      else if (Element.isVariable()) {
        DIVariable DV(Element);
        ElemDie = new DIE(dwarf::DW_TAG_variable);
        addString(ElemDie, dwarf::DW_AT_name, dwarf::DW_FORM_string,
                  DV.getName());
        addType(ElemDie, DV.getType());
        addUInt(ElemDie, dwarf::DW_AT_declaration, dwarf::DW_FORM_flag, 1);
        addUInt(ElemDie, dwarf::DW_AT_external, dwarf::DW_FORM_flag, 1);
        addSourceLine(ElemDie, DV);
      } else if (Element.isDerivedType())
        ElemDie = createMemberDIE(DIDerivedType(Element));
      else
        continue;
      Buffer.addChild(ElemDie);
    }

    if (CTy.isAppleBlockExtension())
      addUInt(&Buffer, dwarf::DW_AT_APPLE_block, dwarf::DW_FORM_flag, 1);

    unsigned RLang = CTy.getRunTimeLang();
    if (RLang)
      addUInt(&Buffer, dwarf::DW_AT_APPLE_runtime_class,
              dwarf::DW_FORM_data1, RLang);

    DICompositeType ContainingType = CTy.getContainingType();
    if (DIDescriptor(ContainingType).isCompositeType())
      addDIEEntry(&Buffer, dwarf::DW_AT_containing_type, dwarf::DW_FORM_ref4,
                  getOrCreateTypeDIE(DIType(ContainingType)));
    else {
      DIDescriptor Context = CTy.getContext();
      addToContextOwner(&Buffer, Context);
    }

    if (CTy.isObjcClassComplete())
      addUInt(&Buffer, dwarf::DW_AT_APPLE_objc_complete_type,
              dwarf::DW_FORM_flag, 1);

    if (Tag == dwarf::DW_TAG_class_type) 
      addTemplateParams(Buffer, CTy.getTemplateParams());

    break;
  }
  default:
    break;
  }

  // Add name if not anonymous or intermediate type.
  if (!Name.empty())
    addString(&Buffer, dwarf::DW_AT_name, dwarf::DW_FORM_string, Name);

  if (Tag == dwarf::DW_TAG_enumeration_type || Tag == dwarf::DW_TAG_class_type
      || Tag == dwarf::DW_TAG_structure_type || Tag == dwarf::DW_TAG_union_type)
    {
    // Add size if non-zero (derived types might be zero-sized.)
    if (Size)
      addUInt(&Buffer, dwarf::DW_AT_byte_size, 0, Size);
    else {
      // Add zero size if it is not a forward declaration.
      if (CTy.isForwardDecl())
        addUInt(&Buffer, dwarf::DW_AT_declaration, dwarf::DW_FORM_flag, 1);
      else
        addUInt(&Buffer, dwarf::DW_AT_byte_size, 0, 0);
    }

    // Add source line info if available.
    if (!CTy.isForwardDecl())
      addSourceLine(&Buffer, CTy);
  }
}

/// getOrCreateTemplateTypeParameterDIE - Find existing DIE or create new DIE 
/// for the given DITemplateTypeParameter.
DIE *
CompileUnit::getOrCreateTemplateTypeParameterDIE(DITemplateTypeParameter TP) {
  DIE *ParamDIE = getDIE(TP);
  if (ParamDIE)
    return ParamDIE;

  ParamDIE = new DIE(dwarf::DW_TAG_template_type_parameter);
  addType(ParamDIE, TP.getType());
  addString(ParamDIE, dwarf::DW_AT_name, dwarf::DW_FORM_string, TP.getName());
  return ParamDIE;
}

/// getOrCreateTemplateValueParameterDIE - Find existing DIE or create new DIE 
/// for the given DITemplateValueParameter.
DIE *
CompileUnit::getOrCreateTemplateValueParameterDIE(DITemplateValueParameter TPV) {
  DIE *ParamDIE = getDIE(TPV);
  if (ParamDIE)
    return ParamDIE;

  ParamDIE = new DIE(dwarf::DW_TAG_template_value_parameter);
  addType(ParamDIE, TPV.getType());
  if (!TPV.getName().empty())
    addString(ParamDIE, dwarf::DW_AT_name, dwarf::DW_FORM_string, TPV.getName());
  addUInt(ParamDIE, dwarf::DW_AT_const_value, dwarf::DW_FORM_udata, 
          TPV.getValue());
  return ParamDIE;
}

/// getOrCreateNameSpace - Create a DIE for DINameSpace.
DIE *CompileUnit::getOrCreateNameSpace(DINameSpace NS) {
  DIE *NDie = getDIE(NS);
  if (NDie)
    return NDie;
  NDie = new DIE(dwarf::DW_TAG_namespace);
  insertDIE(NS, NDie);
  if (!NS.getName().empty())
    addString(NDie, dwarf::DW_AT_name, dwarf::DW_FORM_string, NS.getName());
  addSourceLine(NDie, NS);
  addToContextOwner(NDie, NS.getContext());
  return NDie;
}

/// getRealLinkageName - If special LLVM prefix that is used to inform the asm
/// printer to not emit usual symbol prefix before the symbol name is used then
/// return linkage name after skipping this special LLVM prefix.
static StringRef getRealLinkageName(StringRef LinkageName) {
  char One = '\1';
  if (LinkageName.startswith(StringRef(&One, 1)))
    return LinkageName.substr(1);
  return LinkageName;
}

/// getOrCreateSubprogramDIE - Create new DIE using SP.
DIE *CompileUnit::getOrCreateSubprogramDIE(DISubprogram SP) {
  DIE *SPDie = getDIE(SP);
  if (SPDie)
    return SPDie;

  SPDie = new DIE(dwarf::DW_TAG_subprogram);
  
  // DW_TAG_inlined_subroutine may refer to this DIE.
  insertDIE(SP, SPDie);
  
  // Add to context owner.
  addToContextOwner(SPDie, SP.getContext());

  // Add function template parameters.
  addTemplateParams(*SPDie, SP.getTemplateParams());

  StringRef LinkageName = SP.getLinkageName();
  if (!LinkageName.empty())
    addString(SPDie, dwarf::DW_AT_MIPS_linkage_name, 
                    dwarf::DW_FORM_string,
                    getRealLinkageName(LinkageName));

  // If this DIE is going to refer declaration info using AT_specification
  // then there is no need to add other attributes.
  if (SP.getFunctionDeclaration().isSubprogram())
    return SPDie;

  // Constructors and operators for anonymous aggregates do not have names.
  if (!SP.getName().empty())
    addString(SPDie, dwarf::DW_AT_name, dwarf::DW_FORM_string, 
                    SP.getName());

  addSourceLine(SPDie, SP);

  if (SP.isPrototyped()) 
    addUInt(SPDie, dwarf::DW_AT_prototyped, dwarf::DW_FORM_flag, 1);

  // Add Return Type.
  DICompositeType SPTy = SP.getType();
  DIArray Args = SPTy.getTypeArray();
  unsigned SPTag = SPTy.getTag();

  if (Args.getNumElements() == 0 || SPTag != dwarf::DW_TAG_subroutine_type)
    addType(SPDie, SPTy);
  else
    addType(SPDie, DIType(Args.getElement(0)));

  unsigned VK = SP.getVirtuality();
  if (VK) {
    addUInt(SPDie, dwarf::DW_AT_virtuality, dwarf::DW_FORM_flag, VK);
    DIEBlock *Block = getDIEBlock();
    addUInt(Block, 0, dwarf::DW_FORM_data1, dwarf::DW_OP_constu);
    addUInt(Block, 0, dwarf::DW_FORM_udata, SP.getVirtualIndex());
    addBlock(SPDie, dwarf::DW_AT_vtable_elem_location, 0, Block);
    ContainingTypeMap.insert(std::make_pair(SPDie,
                                            SP.getContainingType()));
  }

  if (!SP.isDefinition()) {
    addUInt(SPDie, dwarf::DW_AT_declaration, dwarf::DW_FORM_flag, 1);
    
    // Add arguments. Do not add arguments for subprogram definition. They will
    // be handled while processing variables.
    DICompositeType SPTy = SP.getType();
    DIArray Args = SPTy.getTypeArray();
    unsigned SPTag = SPTy.getTag();

    if (SPTag == dwarf::DW_TAG_subroutine_type)
      for (unsigned i = 1, N =  Args.getNumElements(); i < N; ++i) {
        DIE *Arg = new DIE(dwarf::DW_TAG_formal_parameter);
        DIType ATy = DIType(DIType(Args.getElement(i)));
        addType(Arg, ATy);
        if (ATy.isArtificial())
          addUInt(Arg, dwarf::DW_AT_artificial, dwarf::DW_FORM_flag, 1);
        SPDie->addChild(Arg);
      }
  }

  if (SP.isArtificial())
    addUInt(SPDie, dwarf::DW_AT_artificial, dwarf::DW_FORM_flag, 1);

  if (!SP.isLocalToUnit())
    addUInt(SPDie, dwarf::DW_AT_external, dwarf::DW_FORM_flag, 1);

  if (SP.isOptimized())
    addUInt(SPDie, dwarf::DW_AT_APPLE_optimized, dwarf::DW_FORM_flag, 1);

  if (unsigned isa = Asm->getISAEncoding()) {
    addUInt(SPDie, dwarf::DW_AT_APPLE_isa, dwarf::DW_FORM_flag, isa);
  }

  return SPDie;
}

// Return const expression if value is a GEP to access merged global
// constant. e.g.
// i8* getelementptr ({ i8, i8, i8, i8 }* @_MergedGlobals, i32 0, i32 0)
static const ConstantExpr *getMergedGlobalExpr(const Value *V) {
  const ConstantExpr *CE = dyn_cast_or_null<ConstantExpr>(V);
  if (!CE || CE->getNumOperands() != 3 ||
      CE->getOpcode() != Instruction::GetElementPtr)
    return NULL;

  // First operand points to a global struct.
  Value *Ptr = CE->getOperand(0);
  if (!isa<GlobalValue>(Ptr) ||
      !isa<StructType>(cast<PointerType>(Ptr->getType())->getElementType()))
    return NULL;

  // Second operand is zero.
  const ConstantInt *CI = dyn_cast_or_null<ConstantInt>(CE->getOperand(1));
  if (!CI || !CI->isZero())
    return NULL;

  // Third operand is offset.
  if (!isa<ConstantInt>(CE->getOperand(2)))
    return NULL;

  return CE;
}

/// createGlobalVariableDIE - create global variable DIE.
void CompileUnit::createGlobalVariableDIE(const MDNode *N) {
  // Check for pre-existence.
  if (getDIE(N))
    return;

  DIGlobalVariable GV(N);
  if (!GV.Verify())
    return;

  DIE *VariableDIE = new DIE(GV.getTag());
  // Add to map.
  insertDIE(N, VariableDIE);

  // Add name.
  addString(VariableDIE, dwarf::DW_AT_name, dwarf::DW_FORM_string,
                   GV.getDisplayName());
  StringRef LinkageName = GV.getLinkageName();
  bool isGlobalVariable = GV.getGlobal() != NULL;
  if (!LinkageName.empty() && isGlobalVariable)
    addString(VariableDIE, dwarf::DW_AT_MIPS_linkage_name, 
                     dwarf::DW_FORM_string,
                     getRealLinkageName(LinkageName));
  // Add type.
  DIType GTy = GV.getType();
  addType(VariableDIE, GTy);

  // Add scoping info.
  if (!GV.isLocalToUnit()) {
    addUInt(VariableDIE, dwarf::DW_AT_external, dwarf::DW_FORM_flag, 1);
    // Expose as global. 
    addGlobal(GV.getName(), VariableDIE);
  }
  // Add line number info.
  addSourceLine(VariableDIE, GV);
  // Add to context owner.
  DIDescriptor GVContext = GV.getContext();
  addToContextOwner(VariableDIE, GVContext);
  // Add location.
  if (isGlobalVariable) {
    DIEBlock *Block = new (DIEValueAllocator) DIEBlock();
    addUInt(Block, 0, dwarf::DW_FORM_data1, dwarf::DW_OP_addr);
    addLabel(Block, 0, dwarf::DW_FORM_udata,
             Asm->Mang->getSymbol(GV.getGlobal()));
    // Do not create specification DIE if context is either compile unit
    // or a subprogram.
    if (GVContext && GV.isDefinition() && !GVContext.isCompileUnit() &&
        !GVContext.isFile() && !isSubprogramContext(GVContext)) {
      // Create specification DIE.
      DIE *VariableSpecDIE = new DIE(dwarf::DW_TAG_variable);
      addDIEEntry(VariableSpecDIE, dwarf::DW_AT_specification,
                  dwarf::DW_FORM_ref4, VariableDIE);
      addBlock(VariableSpecDIE, dwarf::DW_AT_location, 0, Block);
      addUInt(VariableDIE, dwarf::DW_AT_declaration, dwarf::DW_FORM_flag,
                     1);
      addDie(VariableSpecDIE);
    } else {
      addBlock(VariableDIE, dwarf::DW_AT_location, 0, Block);
    } 
  } else if (const ConstantInt *CI = 
             dyn_cast_or_null<ConstantInt>(GV.getConstant()))
    addConstantValue(VariableDIE, CI, GTy.isUnsignedDIType());
  else if (const ConstantExpr *CE = getMergedGlobalExpr(N->getOperand(11))) {
    // GV is a merged global.
    DIEBlock *Block = new (DIEValueAllocator) DIEBlock();
    Value *Ptr = CE->getOperand(0);
    addUInt(Block, 0, dwarf::DW_FORM_data1, dwarf::DW_OP_addr);
    addLabel(Block, 0, dwarf::DW_FORM_udata,
                    Asm->Mang->getSymbol(cast<GlobalValue>(Ptr)));
    addUInt(Block, 0, dwarf::DW_FORM_data1, dwarf::DW_OP_constu);
    SmallVector<Value*, 3> Idx(CE->op_begin()+1, CE->op_end());
    addUInt(Block, 0, dwarf::DW_FORM_udata, 
                   Asm->getTargetData().getIndexedOffset(Ptr->getType(), Idx));
    addUInt(Block, 0, dwarf::DW_FORM_data1, dwarf::DW_OP_plus);
    addBlock(VariableDIE, dwarf::DW_AT_location, 0, Block);
  }

  return;
}

/// constructSubrangeDIE - Construct subrange DIE from DISubrange.
void CompileUnit::constructSubrangeDIE(DIE &Buffer, DISubrange SR, DIE *IndexTy){
  DIE *DW_Subrange = new DIE(dwarf::DW_TAG_subrange_type);
  addDIEEntry(DW_Subrange, dwarf::DW_AT_type, dwarf::DW_FORM_ref4, IndexTy);
  int64_t L = SR.getLo();
  int64_t H = SR.getHi();

  // The L value defines the lower bounds which is typically zero for C/C++. The
  // H value is the upper bounds.  Values are 64 bit.  H - L + 1 is the size
  // of the array. If L > H then do not emit DW_AT_lower_bound and 
  // DW_AT_upper_bound attributes. If L is zero and H is also zero then the
  // array has one element and in such case do not emit lower bound.

  if (L > H) {
    Buffer.addChild(DW_Subrange);
    return;
  }
  if (L)
    addSInt(DW_Subrange, dwarf::DW_AT_lower_bound, 0, L);
  addSInt(DW_Subrange, dwarf::DW_AT_upper_bound, 0, H);
  Buffer.addChild(DW_Subrange);
}

/// constructArrayTypeDIE - Construct array type DIE from DICompositeType.
void CompileUnit::constructArrayTypeDIE(DIE &Buffer,
                                        DICompositeType *CTy) {
  Buffer.setTag(dwarf::DW_TAG_array_type);
  if (CTy->getTag() == dwarf::DW_TAG_vector_type)
    addUInt(&Buffer, dwarf::DW_AT_GNU_vector, dwarf::DW_FORM_flag, 1);

  // Emit derived type.
  addType(&Buffer, CTy->getTypeDerivedFrom());
  DIArray Elements = CTy->getTypeArray();

  // Get an anonymous type for index type.
  DIE *IdxTy = getIndexTyDie();
  if (!IdxTy) {
    // Construct an anonymous type for index type.
    IdxTy = new DIE(dwarf::DW_TAG_base_type);
    addUInt(IdxTy, dwarf::DW_AT_byte_size, 0, sizeof(int32_t));
    addUInt(IdxTy, dwarf::DW_AT_encoding, dwarf::DW_FORM_data1,
            dwarf::DW_ATE_signed);
    addDie(IdxTy);
    setIndexTyDie(IdxTy);
  }

  // Add subranges to array type.
  for (unsigned i = 0, N = Elements.getNumElements(); i < N; ++i) {
    DIDescriptor Element = Elements.getElement(i);
    if (Element.getTag() == dwarf::DW_TAG_subrange_type)
      constructSubrangeDIE(Buffer, DISubrange(Element), IdxTy);
  }
}

/// constructEnumTypeDIE - Construct enum type DIE from DIEnumerator.
DIE *CompileUnit::constructEnumTypeDIE(DIEnumerator ETy) {
  DIE *Enumerator = new DIE(dwarf::DW_TAG_enumerator);
  StringRef Name = ETy.getName();
  addString(Enumerator, dwarf::DW_AT_name, dwarf::DW_FORM_string, Name);
  int64_t Value = ETy.getEnumValue();
  addSInt(Enumerator, dwarf::DW_AT_const_value, dwarf::DW_FORM_sdata, Value);
  return Enumerator;
}

/// constructContainingTypeDIEs - Construct DIEs for types that contain
/// vtables.
void CompileUnit::constructContainingTypeDIEs() {
  for (DenseMap<DIE *, const MDNode *>::iterator CI = ContainingTypeMap.begin(),
         CE = ContainingTypeMap.end(); CI != CE; ++CI) {
    DIE *SPDie = CI->first;
    const MDNode *N = CI->second;
    if (!N) continue;
    DIE *NDie = getDIE(N);
    if (!NDie) continue;
    addDIEEntry(SPDie, dwarf::DW_AT_containing_type, dwarf::DW_FORM_ref4, NDie);
  }
}

/// constructVariableDIE - Construct a DIE for the given DbgVariable.
DIE *CompileUnit::constructVariableDIE(DbgVariable *DV, bool isScopeAbstract) {
  StringRef Name = DV->getName();
  if (Name.empty())
    return NULL;

  // Translate tag to proper Dwarf tag.
  unsigned Tag = DV->getTag();

  // Define variable debug information entry.
  DIE *VariableDie = new DIE(Tag);
  DbgVariable *AbsVar = DV->getAbstractVariable();
  DIE *AbsDIE = AbsVar ? AbsVar->getDIE() : NULL;
  if (AbsDIE)
    addDIEEntry(VariableDie, dwarf::DW_AT_abstract_origin,
                            dwarf::DW_FORM_ref4, AbsDIE);
  else {
    addString(VariableDie, dwarf::DW_AT_name, 
                          dwarf::DW_FORM_string, Name);
    addSourceLine(VariableDie, DV->getVariable());
    addType(VariableDie, DV->getType());
  }

  if (DV->isArtificial())
    addUInt(VariableDie, dwarf::DW_AT_artificial,
                        dwarf::DW_FORM_flag, 1);

  if (isScopeAbstract) {
    DV->setDIE(VariableDie);
    return VariableDie;
  }

  // Add variable address.

  unsigned Offset = DV->getDotDebugLocOffset();
  if (Offset != ~0U) {
    addLabel(VariableDie, dwarf::DW_AT_location,
                         dwarf::DW_FORM_data4,
                         Asm->GetTempSymbol("debug_loc", Offset));
    DV->setDIE(VariableDie);
    return VariableDie;
  }

  // Check if variable is described by a DBG_VALUE instruction.
  if (const MachineInstr *DVInsn = DV->getMInsn()) {
    bool updated = false;
    if (DVInsn->getNumOperands() == 3) {
      if (DVInsn->getOperand(0).isReg()) {
        const MachineOperand RegOp = DVInsn->getOperand(0);
        const TargetRegisterInfo *TRI = Asm->TM.getRegisterInfo();
        if (DVInsn->getOperand(1).isImm() &&
            TRI->getFrameRegister(*Asm->MF) == RegOp.getReg()) {
          unsigned FrameReg = 0;
          const TargetFrameLowering *TFI = Asm->TM.getFrameLowering();
          int Offset = 
            TFI->getFrameIndexReference(*Asm->MF, 
                                        DVInsn->getOperand(1).getImm(), 
                                        FrameReg);
          MachineLocation Location(FrameReg, Offset);
          addVariableAddress(DV, VariableDie, Location);
          
        } else if (RegOp.getReg())
          addVariableAddress(DV, VariableDie, 
                                         MachineLocation(RegOp.getReg()));
        updated = true;
      }
      else if (DVInsn->getOperand(0).isImm())
        updated = 
          addConstantValue(VariableDie, DVInsn->getOperand(0),
                                       DV->getType());
      else if (DVInsn->getOperand(0).isFPImm())
        updated =
          addConstantFPValue(VariableDie, DVInsn->getOperand(0));
      else if (DVInsn->getOperand(0).isCImm())
        updated =
          addConstantValue(VariableDie, 
                                       DVInsn->getOperand(0).getCImm(),
                                       DV->getType().isUnsignedDIType());
    } else {
      addVariableAddress(DV, VariableDie, 
                                     Asm->getDebugValueLocation(DVInsn));
      updated = true;
    }
    if (!updated) {
      // If variableDie is not updated then DBG_VALUE instruction does not
      // have valid variable info.
      delete VariableDie;
      return NULL;
    }
    DV->setDIE(VariableDie);
    return VariableDie;
  } else {
    // .. else use frame index.
    int FI = DV->getFrameIndex();
    if (FI != ~0) {
      unsigned FrameReg = 0;
      const TargetFrameLowering *TFI = Asm->TM.getFrameLowering();
      int Offset = 
        TFI->getFrameIndexReference(*Asm->MF, FI, FrameReg);
      MachineLocation Location(FrameReg, Offset);
      addVariableAddress(DV, VariableDie, Location);
    }
  }

  DV->setDIE(VariableDie);
  return VariableDie;
}

/// createMemberDIE - Create new member DIE.
DIE *CompileUnit::createMemberDIE(DIDerivedType DT) {
  DIE *MemberDie = new DIE(DT.getTag());
  StringRef Name = DT.getName();
  if (!Name.empty())
    addString(MemberDie, dwarf::DW_AT_name, dwarf::DW_FORM_string, Name);

  addType(MemberDie, DT.getTypeDerivedFrom());

  addSourceLine(MemberDie, DT);

  DIEBlock *MemLocationDie = new (DIEValueAllocator) DIEBlock();
  addUInt(MemLocationDie, 0, dwarf::DW_FORM_data1, dwarf::DW_OP_plus_uconst);

  uint64_t Size = DT.getSizeInBits();
  uint64_t FieldSize = DT.getOriginalTypeSize();

  if (Size != FieldSize) {
    // Handle bitfield.
    addUInt(MemberDie, dwarf::DW_AT_byte_size, 0, DT.getOriginalTypeSize()>>3);
    addUInt(MemberDie, dwarf::DW_AT_bit_size, 0, DT.getSizeInBits());

    uint64_t Offset = DT.getOffsetInBits();
    uint64_t AlignMask = ~(DT.getAlignInBits() - 1);
    uint64_t HiMark = (Offset + FieldSize) & AlignMask;
    uint64_t FieldOffset = (HiMark - FieldSize);
    Offset -= FieldOffset;

    // Maybe we need to work from the other end.
    if (Asm->getTargetData().isLittleEndian())
      Offset = FieldSize - (Offset + Size);
    addUInt(MemberDie, dwarf::DW_AT_bit_offset, 0, Offset);

    // Here WD_AT_data_member_location points to the anonymous
    // field that includes this bit field.
    addUInt(MemLocationDie, 0, dwarf::DW_FORM_udata, FieldOffset >> 3);

  } else
    // This is not a bitfield.
    addUInt(MemLocationDie, 0, dwarf::DW_FORM_udata, DT.getOffsetInBits() >> 3);

  if (DT.getTag() == dwarf::DW_TAG_inheritance
      && DT.isVirtual()) {

    // For C++, virtual base classes are not at fixed offset. Use following
    // expression to extract appropriate offset from vtable.
    // BaseAddr = ObAddr + *((*ObAddr) - Offset)

    DIEBlock *VBaseLocationDie = new (DIEValueAllocator) DIEBlock();
    addUInt(VBaseLocationDie, 0, dwarf::DW_FORM_data1, dwarf::DW_OP_dup);
    addUInt(VBaseLocationDie, 0, dwarf::DW_FORM_data1, dwarf::DW_OP_deref);
    addUInt(VBaseLocationDie, 0, dwarf::DW_FORM_data1, dwarf::DW_OP_constu);
    addUInt(VBaseLocationDie, 0, dwarf::DW_FORM_udata, DT.getOffsetInBits());
    addUInt(VBaseLocationDie, 0, dwarf::DW_FORM_data1, dwarf::DW_OP_minus);
    addUInt(VBaseLocationDie, 0, dwarf::DW_FORM_data1, dwarf::DW_OP_deref);
    addUInt(VBaseLocationDie, 0, dwarf::DW_FORM_data1, dwarf::DW_OP_plus);

    addBlock(MemberDie, dwarf::DW_AT_data_member_location, 0,
             VBaseLocationDie);
  } else
    addBlock(MemberDie, dwarf::DW_AT_data_member_location, 0, MemLocationDie);

  if (DT.isProtected())
    addUInt(MemberDie, dwarf::DW_AT_accessibility, dwarf::DW_FORM_flag,
            dwarf::DW_ACCESS_protected);
  else if (DT.isPrivate())
    addUInt(MemberDie, dwarf::DW_AT_accessibility, dwarf::DW_FORM_flag,
            dwarf::DW_ACCESS_private);
  // Otherwise C++ member and base classes are considered public.
  else 
    addUInt(MemberDie, dwarf::DW_AT_accessibility, dwarf::DW_FORM_flag,
            dwarf::DW_ACCESS_public);
  if (DT.isVirtual())
    addUInt(MemberDie, dwarf::DW_AT_virtuality, dwarf::DW_FORM_flag,
            dwarf::DW_VIRTUALITY_virtual);

  // Objective-C properties.
  StringRef PropertyName = DT.getObjCPropertyName();
  if (!PropertyName.empty()) {
    addString(MemberDie, dwarf::DW_AT_APPLE_property_name, dwarf::DW_FORM_string,
              PropertyName);
    StringRef GetterName = DT.getObjCPropertyGetterName();
    if (!GetterName.empty())
      addString(MemberDie, dwarf::DW_AT_APPLE_property_getter,
                dwarf::DW_FORM_string, GetterName);
    StringRef SetterName = DT.getObjCPropertySetterName();
    if (!SetterName.empty())
      addString(MemberDie, dwarf::DW_AT_APPLE_property_setter,
                dwarf::DW_FORM_string, SetterName);
    unsigned PropertyAttributes = 0;
    if (DT.isReadOnlyObjCProperty())
      PropertyAttributes |= dwarf::DW_APPLE_PROPERTY_readonly;
    if (DT.isReadWriteObjCProperty())
      PropertyAttributes |= dwarf::DW_APPLE_PROPERTY_readwrite;
    if (DT.isAssignObjCProperty())
      PropertyAttributes |= dwarf::DW_APPLE_PROPERTY_assign;
    if (DT.isRetainObjCProperty())
      PropertyAttributes |= dwarf::DW_APPLE_PROPERTY_retain;
    if (DT.isCopyObjCProperty())
      PropertyAttributes |= dwarf::DW_APPLE_PROPERTY_copy;
    if (DT.isNonAtomicObjCProperty())
      PropertyAttributes |= dwarf::DW_APPLE_PROPERTY_nonatomic;
    if (PropertyAttributes)
      addUInt(MemberDie, dwarf::DW_AT_APPLE_property_attribute, 0, 
              PropertyAttributes);
  }
  return MemberDie;
}
