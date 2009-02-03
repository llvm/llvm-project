//===-- LLParser.cpp - Parser Class ---------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the parser class for .ll files.
//
//===----------------------------------------------------------------------===//

#include "LLParser.h"
#include "llvm/AutoUpgrade.h"
#include "llvm/CallingConv.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/InlineAsm.h"
#include "llvm/Instructions.h"
#include "llvm/Module.h"
#include "llvm/ValueSymbolTable.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

namespace llvm {
  /// ValID - Represents a reference of a definition of some sort with no type.
  /// There are several cases where we have to parse the value but where the
  /// type can depend on later context.  This may either be a numeric reference
  /// or a symbolic (%var) reference.  This is just a discriminated union.
  struct ValID {
    enum {
      t_LocalID, t_GlobalID,      // ID in UIntVal.
      t_LocalName, t_GlobalName,  // Name in StrVal.
      t_APSInt, t_APFloat,        // Value in APSIntVal/APFloatVal.
      t_Null, t_Undef, t_Zero,    // No value.
      t_EmptyArray,               // No value:  []
      t_Constant,                 // Value in ConstantVal.
      t_InlineAsm                 // Value in StrVal/StrVal2/UIntVal.
    } Kind;
    
    LLParser::LocTy Loc;
    unsigned UIntVal;
    std::string StrVal, StrVal2;
    APSInt APSIntVal;
    APFloat APFloatVal;
    Constant *ConstantVal;
    ValID() : APFloatVal(0.0) {}
  };
}

/// Run: module ::= toplevelentity*
bool LLParser::Run() {
  // Prime the lexer.
  Lex.Lex();

  return ParseTopLevelEntities() ||
         ValidateEndOfModule();
}

/// ValidateEndOfModule - Do final validity and sanity checks at the end of the
/// module.
bool LLParser::ValidateEndOfModule() {
  if (!ForwardRefTypes.empty())
    return Error(ForwardRefTypes.begin()->second.second,
                 "use of undefined type named '" +
                 ForwardRefTypes.begin()->first + "'");
  if (!ForwardRefTypeIDs.empty())
    return Error(ForwardRefTypeIDs.begin()->second.second,
                 "use of undefined type '%" +
                 utostr(ForwardRefTypeIDs.begin()->first) + "'");
  
  if (!ForwardRefVals.empty())
    return Error(ForwardRefVals.begin()->second.second,
                 "use of undefined value '@" + ForwardRefVals.begin()->first +
                 "'");
  
  if (!ForwardRefValIDs.empty())
    return Error(ForwardRefValIDs.begin()->second.second,
                 "use of undefined value '@" +
                 utostr(ForwardRefValIDs.begin()->first) + "'");
  
  // Look for intrinsic functions and CallInst that need to be upgraded
  for (Module::iterator FI = M->begin(), FE = M->end(); FI != FE; )
    UpgradeCallsToIntrinsic(FI++); // must be post-increment, as we remove
  
  return false;
}

//===----------------------------------------------------------------------===//
// Top-Level Entities
//===----------------------------------------------------------------------===//

bool LLParser::ParseTopLevelEntities() {
  while (1) {
    switch (Lex.getKind()) {
    default:         return TokError("expected top-level entity");
    case lltok::Eof: return false;
    //case lltok::kw_define:
    case lltok::kw_declare: if (ParseDeclare()) return true; break;
    case lltok::kw_define:  if (ParseDefine()) return true; break;
    case lltok::kw_module:  if (ParseModuleAsm()) return true; break;
    case lltok::kw_target:  if (ParseTargetDefinition()) return true; break;
    case lltok::kw_deplibs: if (ParseDepLibs()) return true; break;
    case lltok::kw_type:    if (ParseUnnamedType()) return true; break;
    case lltok::StringConstant: // FIXME: REMOVE IN LLVM 3.0
    case lltok::LocalVar:   if (ParseNamedType()) return true; break;
    case lltok::GlobalVar:  if (ParseNamedGlobal()) return true; break;

    // The Global variable production with no name can have many different
    // optional leading prefixes, the production is:
    // GlobalVar ::= OptionalLinkage OptionalVisibility OptionalThreadLocal
    //               OptionalAddrSpace ('constant'|'global') ...
    case lltok::kw_private:       // OptionalLinkage
    case lltok::kw_internal:      // OptionalLinkage
    case lltok::kw_weak:          // OptionalLinkage
    case lltok::kw_linkonce:      // OptionalLinkage
    case lltok::kw_appending:     // OptionalLinkage
    case lltok::kw_dllexport:     // OptionalLinkage
    case lltok::kw_common:        // OptionalLinkage
    case lltok::kw_dllimport:     // OptionalLinkage
    case lltok::kw_extern_weak:   // OptionalLinkage
    case lltok::kw_external: {    // OptionalLinkage
      unsigned Linkage, Visibility;
      if (ParseOptionalLinkage(Linkage) ||
          ParseOptionalVisibility(Visibility) ||
          ParseGlobal("", 0, Linkage, true, Visibility))
        return true;
      break;
    }
    case lltok::kw_default:       // OptionalVisibility
    case lltok::kw_hidden:        // OptionalVisibility
    case lltok::kw_protected: {   // OptionalVisibility
      unsigned Visibility;
      if (ParseOptionalVisibility(Visibility) ||
          ParseGlobal("", 0, 0, false, Visibility))
        return true;
      break;
    }
        
    case lltok::kw_thread_local:  // OptionalThreadLocal
    case lltok::kw_addrspace:     // OptionalAddrSpace
    case lltok::kw_constant:      // GlobalType
    case lltok::kw_global:        // GlobalType
      if (ParseGlobal("", 0, 0, false, 0)) return true;
      break;
    }
  }
}


/// toplevelentity
///   ::= 'module' 'asm' STRINGCONSTANT
bool LLParser::ParseModuleAsm() {
  assert(Lex.getKind() == lltok::kw_module);
  Lex.Lex();
  
  std::string AsmStr; 
  if (ParseToken(lltok::kw_asm, "expected 'module asm'") ||
      ParseStringConstant(AsmStr)) return true;
  
  const std::string &AsmSoFar = M->getModuleInlineAsm();
  if (AsmSoFar.empty())
    M->setModuleInlineAsm(AsmStr);
  else
    M->setModuleInlineAsm(AsmSoFar+"\n"+AsmStr);
  return false;
}

/// toplevelentity
///   ::= 'target' 'triple' '=' STRINGCONSTANT
///   ::= 'target' 'datalayout' '=' STRINGCONSTANT
bool LLParser::ParseTargetDefinition() {
  assert(Lex.getKind() == lltok::kw_target);
  std::string Str;
  switch (Lex.Lex()) {
  default: return TokError("unknown target property");
  case lltok::kw_triple:
    Lex.Lex();
    if (ParseToken(lltok::equal, "expected '=' after target triple") ||
        ParseStringConstant(Str))
      return true;
    M->setTargetTriple(Str);
    return false;
  case lltok::kw_datalayout:
    Lex.Lex();
    if (ParseToken(lltok::equal, "expected '=' after target datalayout") ||
        ParseStringConstant(Str))
      return true;
    M->setDataLayout(Str);
    return false;
  }
}

/// toplevelentity
///   ::= 'deplibs' '=' '[' ']'
///   ::= 'deplibs' '=' '[' STRINGCONSTANT (',' STRINGCONSTANT)* ']'
bool LLParser::ParseDepLibs() {
  assert(Lex.getKind() == lltok::kw_deplibs);
  Lex.Lex();
  if (ParseToken(lltok::equal, "expected '=' after deplibs") ||
      ParseToken(lltok::lsquare, "expected '=' after deplibs"))
    return true;

  if (EatIfPresent(lltok::rsquare))
    return false;
  
  std::string Str;
  if (ParseStringConstant(Str)) return true;
  M->addLibrary(Str);

  while (EatIfPresent(lltok::comma)) {
    if (ParseStringConstant(Str)) return true;
    M->addLibrary(Str);
  }

  return ParseToken(lltok::rsquare, "expected ']' at end of list");
}

/// toplevelentity
///   ::= 'type' type
bool LLParser::ParseUnnamedType() {
  assert(Lex.getKind() == lltok::kw_type);
  LocTy TypeLoc = Lex.getLoc();
  Lex.Lex(); // eat kw_type

  PATypeHolder Ty(Type::VoidTy);
  if (ParseType(Ty)) return true;
 
  unsigned TypeID = NumberedTypes.size();
  
  // We don't allow assigning names to void type
  if (Ty == Type::VoidTy)
    return Error(TypeLoc, "can't assign name to the void type");
  
  // See if this type was previously referenced.
  std::map<unsigned, std::pair<PATypeHolder, LocTy> >::iterator
    FI = ForwardRefTypeIDs.find(TypeID);
  if (FI != ForwardRefTypeIDs.end()) {
    if (FI->second.first.get() == Ty)
      return Error(TypeLoc, "self referential type is invalid");
    
    cast<DerivedType>(FI->second.first.get())->refineAbstractTypeTo(Ty);
    Ty = FI->second.first.get();
    ForwardRefTypeIDs.erase(FI);
  }
  
  NumberedTypes.push_back(Ty);
  
  return false;
}

/// toplevelentity
///   ::= LocalVar '=' 'type' type
bool LLParser::ParseNamedType() {
  std::string Name = Lex.getStrVal();
  LocTy NameLoc = Lex.getLoc();
  Lex.Lex();  // eat LocalVar.
  
  PATypeHolder Ty(Type::VoidTy);
  
  if (ParseToken(lltok::equal, "expected '=' after name") ||
      ParseToken(lltok::kw_type, "expected 'type' after name") ||
      ParseType(Ty))
    return true;
  
  // We don't allow assigning names to void type
  if (Ty == Type::VoidTy)
    return Error(NameLoc, "can't assign name '" + Name + "' to the void type");

  // Set the type name, checking for conflicts as we do so.
  bool AlreadyExists = M->addTypeName(Name, Ty);
  if (!AlreadyExists) return false;

  // See if this type is a forward reference.  We need to eagerly resolve
  // types to allow recursive type redefinitions below.
  std::map<std::string, std::pair<PATypeHolder, LocTy> >::iterator
  FI = ForwardRefTypes.find(Name);
  if (FI != ForwardRefTypes.end()) {
    if (FI->second.first.get() == Ty)
      return Error(NameLoc, "self referential type is invalid");

    cast<DerivedType>(FI->second.first.get())->refineAbstractTypeTo(Ty);
    Ty = FI->second.first.get();
    ForwardRefTypes.erase(FI);
  }
  
  // Inserting a name that is already defined, get the existing name.
  const Type *Existing = M->getTypeByName(Name);
  assert(Existing && "Conflict but no matching type?!");
    
  // Otherwise, this is an attempt to redefine a type. That's okay if
  // the redefinition is identical to the original.
  // FIXME: REMOVE REDEFINITIONS IN LLVM 3.0
  if (Existing == Ty) return false;
  
  // Any other kind of (non-equivalent) redefinition is an error.
  return Error(NameLoc, "redefinition of type named '" + Name + "' of type '" +
               Ty->getDescription() + "'");
}


/// toplevelentity
///   ::= 'declare' FunctionHeader
bool LLParser::ParseDeclare() {
  assert(Lex.getKind() == lltok::kw_declare);
  Lex.Lex();
  
  Function *F;
  return ParseFunctionHeader(F, false);
}

/// toplevelentity
///   ::= 'define' FunctionHeader '{' ...
bool LLParser::ParseDefine() {
  assert(Lex.getKind() == lltok::kw_define);
  Lex.Lex();
  
  Function *F;
  return ParseFunctionHeader(F, true) ||
         ParseFunctionBody(*F);
}

/// ParseGlobalType
///   ::= 'constant'
///   ::= 'global'
bool LLParser::ParseGlobalType(bool &IsConstant) {
  if (Lex.getKind() == lltok::kw_constant)
    IsConstant = true;
  else if (Lex.getKind() == lltok::kw_global)
    IsConstant = false;
  else
    return TokError("expected 'global' or 'constant'");
  Lex.Lex();
  return false;
}

/// ParseNamedGlobal:
///   GlobalVar '=' OptionalVisibility ALIAS ...
///   GlobalVar '=' OptionalLinkage OptionalVisibility ...   -> global variable
bool LLParser::ParseNamedGlobal() {
  assert(Lex.getKind() == lltok::GlobalVar);
  LocTy NameLoc = Lex.getLoc();
  std::string Name = Lex.getStrVal();
  Lex.Lex();
  
  bool HasLinkage;
  unsigned Linkage, Visibility;
  if (ParseToken(lltok::equal, "expected '=' in global variable") ||
      ParseOptionalLinkage(Linkage, HasLinkage) ||
      ParseOptionalVisibility(Visibility))
    return true;
  
  if (HasLinkage || Lex.getKind() != lltok::kw_alias)
    return ParseGlobal(Name, NameLoc, Linkage, HasLinkage, Visibility);
  return ParseAlias(Name, NameLoc, Visibility);
}

/// ParseAlias:
///   ::= GlobalVar '=' OptionalVisibility 'alias' OptionalLinkage Aliasee
/// Aliasee
///   ::= TypeAndValue | 'bitcast' '(' TypeAndValue 'to' Type ')'
///
/// Everything through visibility has already been parsed.
///
bool LLParser::ParseAlias(const std::string &Name, LocTy NameLoc,
                          unsigned Visibility) {
  assert(Lex.getKind() == lltok::kw_alias);
  Lex.Lex();
  unsigned Linkage;
  LocTy LinkageLoc = Lex.getLoc();
  if (ParseOptionalLinkage(Linkage))
    return true;

  if (Linkage != GlobalValue::ExternalLinkage &&
      Linkage != GlobalValue::WeakLinkage &&
      Linkage != GlobalValue::InternalLinkage &&
      Linkage != GlobalValue::PrivateLinkage)
    return Error(LinkageLoc, "invalid linkage type for alias");
  
  Constant *Aliasee;
  LocTy AliaseeLoc = Lex.getLoc();
  if (Lex.getKind() != lltok::kw_bitcast) {
    if (ParseGlobalTypeAndValue(Aliasee)) return true;
  } else {
    // The bitcast dest type is not present, it is implied by the dest type.
    ValID ID;
    if (ParseValID(ID)) return true;
    if (ID.Kind != ValID::t_Constant)
      return Error(AliaseeLoc, "invalid aliasee");
    Aliasee = ID.ConstantVal;
  }
  
  if (!isa<PointerType>(Aliasee->getType()))
    return Error(AliaseeLoc, "alias must have pointer type");

  // Okay, create the alias but do not insert it into the module yet.
  GlobalAlias* GA = new GlobalAlias(Aliasee->getType(),
                                    (GlobalValue::LinkageTypes)Linkage, Name,
                                    Aliasee);
  GA->setVisibility((GlobalValue::VisibilityTypes)Visibility);
  
  // See if this value already exists in the symbol table.  If so, it is either
  // a redefinition or a definition of a forward reference.
  if (GlobalValue *Val =
        cast_or_null<GlobalValue>(M->getValueSymbolTable().lookup(Name))) {
    // See if this was a redefinition.  If so, there is no entry in
    // ForwardRefVals.
    std::map<std::string, std::pair<GlobalValue*, LocTy> >::iterator
      I = ForwardRefVals.find(Name);
    if (I == ForwardRefVals.end())
      return Error(NameLoc, "redefinition of global named '@" + Name + "'");

    // Otherwise, this was a definition of forward ref.  Verify that types
    // agree.
    if (Val->getType() != GA->getType())
      return Error(NameLoc,
              "forward reference and definition of alias have different types");
    
    // If they agree, just RAUW the old value with the alias and remove the
    // forward ref info.
    Val->replaceAllUsesWith(GA);
    Val->eraseFromParent();
    ForwardRefVals.erase(I);
  }
  
  // Insert into the module, we know its name won't collide now.
  M->getAliasList().push_back(GA);
  assert(GA->getNameStr() == Name && "Should not be a name conflict!");
  
  return false;
}

/// ParseGlobal
///   ::= GlobalVar '=' OptionalLinkage OptionalVisibility OptionalThreadLocal
///       OptionalAddrSpace GlobalType Type Const
///   ::= OptionalLinkage OptionalVisibility OptionalThreadLocal
///       OptionalAddrSpace GlobalType Type Const
///
/// Everything through visibility has been parsed already.
///
bool LLParser::ParseGlobal(const std::string &Name, LocTy NameLoc,
                           unsigned Linkage, bool HasLinkage,
                           unsigned Visibility) {
  unsigned AddrSpace;
  bool ThreadLocal, IsConstant;
  LocTy TyLoc;
    
  PATypeHolder Ty(Type::VoidTy);
  if (ParseOptionalToken(lltok::kw_thread_local, ThreadLocal) ||
      ParseOptionalAddrSpace(AddrSpace) ||
      ParseGlobalType(IsConstant) ||
      ParseType(Ty, TyLoc))
    return true;
  
  // If the linkage is specified and is external, then no initializer is
  // present.
  Constant *Init = 0;
  if (!HasLinkage || (Linkage != GlobalValue::DLLImportLinkage &&
                      Linkage != GlobalValue::ExternalWeakLinkage &&
                      Linkage != GlobalValue::ExternalLinkage)) {
    if (ParseGlobalValue(Ty, Init))
      return true;
  }

  if (isa<FunctionType>(Ty) || Ty == Type::LabelTy)
    return Error(TyLoc, "invald type for global variable");
  
  GlobalVariable *GV = 0;

  // See if the global was forward referenced, if so, use the global.
  if (!Name.empty()) {
    if ((GV = M->getGlobalVariable(Name, true)) &&
        !ForwardRefVals.erase(Name))
      return Error(NameLoc, "redefinition of global '@" + Name + "'");
  } else {
    std::map<unsigned, std::pair<GlobalValue*, LocTy> >::iterator
      I = ForwardRefValIDs.find(NumberedVals.size());
    if (I != ForwardRefValIDs.end()) {
      GV = cast<GlobalVariable>(I->second.first);
      ForwardRefValIDs.erase(I);
    }
  }

  if (GV == 0) {
    GV = new GlobalVariable(Ty, false, GlobalValue::ExternalLinkage, 0, Name,
                            M, false, AddrSpace);
  } else {
    if (GV->getType()->getElementType() != Ty)
      return Error(TyLoc,
            "forward reference and definition of global have different types");
    
    // Move the forward-reference to the correct spot in the module.
    M->getGlobalList().splice(M->global_end(), M->getGlobalList(), GV);
  }

  if (Name.empty())
    NumberedVals.push_back(GV);
  
  // Set the parsed properties on the global.
  if (Init)
    GV->setInitializer(Init);
  GV->setConstant(IsConstant);
  GV->setLinkage((GlobalValue::LinkageTypes)Linkage);
  GV->setVisibility((GlobalValue::VisibilityTypes)Visibility);
  GV->setThreadLocal(ThreadLocal);
  
  // Parse attributes on the global.
  while (Lex.getKind() == lltok::comma) {
    Lex.Lex();
    
    if (Lex.getKind() == lltok::kw_section) {
      Lex.Lex();
      GV->setSection(Lex.getStrVal());
      if (ParseToken(lltok::StringConstant, "expected global section string"))
        return true;
    } else if (Lex.getKind() == lltok::kw_align) {
      unsigned Alignment;
      if (ParseOptionalAlignment(Alignment)) return true;
      GV->setAlignment(Alignment);
    } else {
      TokError("unknown global variable property!");
    }
  }
  
  return false;
}


//===----------------------------------------------------------------------===//
// GlobalValue Reference/Resolution Routines.
//===----------------------------------------------------------------------===//

/// GetGlobalVal - Get a value with the specified name or ID, creating a
/// forward reference record if needed.  This can return null if the value
/// exists but does not have the right type.
GlobalValue *LLParser::GetGlobalVal(const std::string &Name, const Type *Ty,
                                    LocTy Loc) {
  const PointerType *PTy = dyn_cast<PointerType>(Ty);
  if (PTy == 0) {
    Error(Loc, "global variable reference must have pointer type");
    return 0;
  }
  
  // Look this name up in the normal function symbol table.
  GlobalValue *Val =
    cast_or_null<GlobalValue>(M->getValueSymbolTable().lookup(Name));
  
  // If this is a forward reference for the value, see if we already created a
  // forward ref record.
  if (Val == 0) {
    std::map<std::string, std::pair<GlobalValue*, LocTy> >::iterator
      I = ForwardRefVals.find(Name);
    if (I != ForwardRefVals.end())
      Val = I->second.first;
  }
  
  // If we have the value in the symbol table or fwd-ref table, return it.
  if (Val) {
    if (Val->getType() == Ty) return Val;
    Error(Loc, "'@" + Name + "' defined with type '" +
          Val->getType()->getDescription() + "'");
    return 0;
  }
  
  // Otherwise, create a new forward reference for this value and remember it.
  GlobalValue *FwdVal;
  if (const FunctionType *FT = dyn_cast<FunctionType>(PTy->getElementType())) {
    // Function types can return opaque but functions can't.
    if (isa<OpaqueType>(FT->getReturnType())) {
      Error(Loc, "function may not return opaque type");
      return 0;
    }
    
    FwdVal = Function::Create(FT, GlobalValue::ExternalWeakLinkage, Name, M);
  } else {
    FwdVal = new GlobalVariable(PTy->getElementType(), false,
                                GlobalValue::ExternalWeakLinkage, 0, Name, M);
  }
  
  ForwardRefVals[Name] = std::make_pair(FwdVal, Loc);
  return FwdVal;
}

GlobalValue *LLParser::GetGlobalVal(unsigned ID, const Type *Ty, LocTy Loc) {
  const PointerType *PTy = dyn_cast<PointerType>(Ty);
  if (PTy == 0) {
    Error(Loc, "global variable reference must have pointer type");
    return 0;
  }
  
  GlobalValue *Val = ID < NumberedVals.size() ? NumberedVals[ID] : 0;
  
  // If this is a forward reference for the value, see if we already created a
  // forward ref record.
  if (Val == 0) {
    std::map<unsigned, std::pair<GlobalValue*, LocTy> >::iterator
      I = ForwardRefValIDs.find(ID);
    if (I != ForwardRefValIDs.end())
      Val = I->second.first;
  }
  
  // If we have the value in the symbol table or fwd-ref table, return it.
  if (Val) {
    if (Val->getType() == Ty) return Val;
    Error(Loc, "'@" + utostr(ID) + "' defined with type '" +
          Val->getType()->getDescription() + "'");
    return 0;
  }
  
  // Otherwise, create a new forward reference for this value and remember it.
  GlobalValue *FwdVal;
  if (const FunctionType *FT = dyn_cast<FunctionType>(PTy->getElementType())) {
    // Function types can return opaque but functions can't.
    if (isa<OpaqueType>(FT->getReturnType())) {
      Error(Loc, "function may not return opaque type");
      return 0;
    }
    FwdVal = Function::Create(FT, GlobalValue::ExternalWeakLinkage, "", M);
  } else {
    FwdVal = new GlobalVariable(PTy->getElementType(), false,
                                GlobalValue::ExternalWeakLinkage, 0, "", M);
  }
  
  ForwardRefValIDs[ID] = std::make_pair(FwdVal, Loc);
  return FwdVal;
}


//===----------------------------------------------------------------------===//
// Helper Routines.
//===----------------------------------------------------------------------===//

/// ParseToken - If the current token has the specified kind, eat it and return
/// success.  Otherwise, emit the specified error and return failure.
bool LLParser::ParseToken(lltok::Kind T, const char *ErrMsg) {
  if (Lex.getKind() != T)
    return TokError(ErrMsg);
  Lex.Lex();
  return false;
}

/// ParseStringConstant
///   ::= StringConstant
bool LLParser::ParseStringConstant(std::string &Result) {
  if (Lex.getKind() != lltok::StringConstant)
    return TokError("expected string constant");
  Result = Lex.getStrVal();
  Lex.Lex();
  return false;
}

/// ParseUInt32
///   ::= uint32
bool LLParser::ParseUInt32(unsigned &Val) {
  if (Lex.getKind() != lltok::APSInt || Lex.getAPSIntVal().isSigned())
    return TokError("expected integer");
  uint64_t Val64 = Lex.getAPSIntVal().getLimitedValue(0xFFFFFFFFULL+1);
  if (Val64 != unsigned(Val64))
    return TokError("expected 32-bit integer (too large)");
  Val = Val64;
  Lex.Lex();
  return false;
}


/// ParseOptionalAddrSpace
///   := /*empty*/
///   := 'addrspace' '(' uint32 ')'
bool LLParser::ParseOptionalAddrSpace(unsigned &AddrSpace) {
  AddrSpace = 0;
  if (!EatIfPresent(lltok::kw_addrspace))
    return false;
  return ParseToken(lltok::lparen, "expected '(' in address space") ||
         ParseUInt32(AddrSpace) ||
         ParseToken(lltok::rparen, "expected ')' in address space");
}  

/// ParseOptionalAttrs - Parse a potentially empty attribute list.  AttrKind
/// indicates what kind of attribute list this is: 0: function arg, 1: result,
/// 2: function attr.
bool LLParser::ParseOptionalAttrs(unsigned &Attrs, unsigned AttrKind) {
  Attrs = Attribute::None;
  LocTy AttrLoc = Lex.getLoc();
  
  while (1) {
    switch (Lex.getKind()) {
    case lltok::kw_sext:
    case lltok::kw_zext:
      // Treat these as signext/zeroext unless they are function attrs.
      // FIXME: REMOVE THIS IN LLVM 3.0
      if (AttrKind != 2) {
        if (Lex.getKind() == lltok::kw_sext)
          Attrs |= Attribute::SExt;
        else
          Attrs |= Attribute::ZExt;
        break;
      }
      // FALL THROUGH.
    default:  // End of attributes.
      if (AttrKind != 2 && (Attrs & Attribute::FunctionOnly))
        return Error(AttrLoc, "invalid use of function-only attribute");
        
      if (AttrKind != 0 && (Attrs & Attribute::ParameterOnly))
        return Error(AttrLoc, "invalid use of parameter-only attribute");
        
      return false;
    case lltok::kw_zeroext:      Attrs |= Attribute::ZExt; break;
    case lltok::kw_signext:      Attrs |= Attribute::SExt; break;
    case lltok::kw_inreg:        Attrs |= Attribute::InReg; break;
    case lltok::kw_sret:         Attrs |= Attribute::StructRet; break;
    case lltok::kw_noalias:      Attrs |= Attribute::NoAlias; break;
    case lltok::kw_nocapture:    Attrs |= Attribute::NoCapture; break;
    case lltok::kw_byval:        Attrs |= Attribute::ByVal; break;
    case lltok::kw_nest:         Attrs |= Attribute::Nest; break;

    case lltok::kw_noreturn:     Attrs |= Attribute::NoReturn; break;
    case lltok::kw_nounwind:     Attrs |= Attribute::NoUnwind; break;
    case lltok::kw_noinline:     Attrs |= Attribute::NoInline; break;
    case lltok::kw_readnone:     Attrs |= Attribute::ReadNone; break;
    case lltok::kw_readonly:     Attrs |= Attribute::ReadOnly; break;
    case lltok::kw_alwaysinline: Attrs |= Attribute::AlwaysInline; break;
    case lltok::kw_optsize:      Attrs |= Attribute::OptimizeForSize; break;
    case lltok::kw_ssp:          Attrs |= Attribute::StackProtect; break;
    case lltok::kw_sspreq:       Attrs |= Attribute::StackProtectReq; break;

        
    case lltok::kw_align: {
      unsigned Alignment;
      if (ParseOptionalAlignment(Alignment))
        return true;
      Attrs |= Attribute::constructAlignmentFromInt(Alignment);
      continue;
    }
    }
    Lex.Lex();
  }
}

/// ParseOptionalLinkage
///   ::= /*empty*/
///   ::= 'private'
///   ::= 'internal'
///   ::= 'weak'
///   ::= 'linkonce'
///   ::= 'appending'
///   ::= 'dllexport'
///   ::= 'common'
///   ::= 'dllimport'
///   ::= 'extern_weak'
///   ::= 'external'
bool LLParser::ParseOptionalLinkage(unsigned &Res, bool &HasLinkage) {
  HasLinkage = false;
  switch (Lex.getKind()) {
  default:                    Res = GlobalValue::ExternalLinkage; return false;
  case lltok::kw_private:     Res = GlobalValue::PrivateLinkage; break;
  case lltok::kw_internal:    Res = GlobalValue::InternalLinkage; break;
  case lltok::kw_weak:        Res = GlobalValue::WeakLinkage; break;
  case lltok::kw_linkonce:    Res = GlobalValue::LinkOnceLinkage; break;
  case lltok::kw_appending:   Res = GlobalValue::AppendingLinkage; break;
  case lltok::kw_dllexport:   Res = GlobalValue::DLLExportLinkage; break;
  case lltok::kw_common:      Res = GlobalValue::CommonLinkage; break;
  case lltok::kw_dllimport:   Res = GlobalValue::DLLImportLinkage; break;
  case lltok::kw_extern_weak: Res = GlobalValue::ExternalWeakLinkage; break;
  case lltok::kw_external:    Res = GlobalValue::ExternalLinkage; break;
  }
  Lex.Lex();
  HasLinkage = true;
  return false;
}

/// ParseOptionalVisibility
///   ::= /*empty*/
///   ::= 'default'
///   ::= 'hidden'
///   ::= 'protected'
/// 
bool LLParser::ParseOptionalVisibility(unsigned &Res) {
  switch (Lex.getKind()) {
  default:                  Res = GlobalValue::DefaultVisibility; return false;
  case lltok::kw_default:   Res = GlobalValue::DefaultVisibility; break;
  case lltok::kw_hidden:    Res = GlobalValue::HiddenVisibility; break;
  case lltok::kw_protected: Res = GlobalValue::ProtectedVisibility; break;
  }
  Lex.Lex();
  return false;
}

/// ParseOptionalCallingConv
///   ::= /*empty*/
///   ::= 'ccc'
///   ::= 'fastcc'
///   ::= 'coldcc'
///   ::= 'x86_stdcallcc'
///   ::= 'x86_fastcallcc'
///   ::= 'cc' UINT
/// 
bool LLParser::ParseOptionalCallingConv(unsigned &CC) {
  switch (Lex.getKind()) {
  default:                       CC = CallingConv::C; return false;
  case lltok::kw_ccc:            CC = CallingConv::C; break;
  case lltok::kw_fastcc:         CC = CallingConv::Fast; break;
  case lltok::kw_coldcc:         CC = CallingConv::Cold; break;
  case lltok::kw_x86_stdcallcc:  CC = CallingConv::X86_StdCall; break;
  case lltok::kw_x86_fastcallcc: CC = CallingConv::X86_FastCall; break;
  case lltok::kw_cc:             Lex.Lex(); return ParseUInt32(CC);
  }
  Lex.Lex();
  return false;
}

/// ParseOptionalAlignment
///   ::= /* empty */
///   ::= 'align' 4
bool LLParser::ParseOptionalAlignment(unsigned &Alignment) {
  Alignment = 0;
  if (!EatIfPresent(lltok::kw_align))
    return false;
  LocTy AlignLoc = Lex.getLoc();
  if (ParseUInt32(Alignment)) return true;
  if (!isPowerOf2_32(Alignment))
    return Error(AlignLoc, "alignment is not a power of two");
  return false;
}

/// ParseOptionalCommaAlignment
///   ::= /* empty */
///   ::= ',' 'align' 4
bool LLParser::ParseOptionalCommaAlignment(unsigned &Alignment) {
  Alignment = 0;
  if (!EatIfPresent(lltok::comma))
    return false;
  return ParseToken(lltok::kw_align, "expected 'align'") ||
         ParseUInt32(Alignment);
}

/// ParseIndexList
///    ::=  (',' uint32)+
bool LLParser::ParseIndexList(SmallVectorImpl<unsigned> &Indices) {
  if (Lex.getKind() != lltok::comma)
    return TokError("expected ',' as start of index list");
  
  while (EatIfPresent(lltok::comma)) {
    unsigned Idx;
    if (ParseUInt32(Idx)) return true;
    Indices.push_back(Idx);
  }
  
  return false;
}

//===----------------------------------------------------------------------===//
// Type Parsing.
//===----------------------------------------------------------------------===//

/// ParseType - Parse and resolve a full type.
bool LLParser::ParseType(PATypeHolder &Result) {
  if (ParseTypeRec(Result)) return true;
  
  // Verify no unresolved uprefs.
  if (!UpRefs.empty())
    return Error(UpRefs.back().Loc, "invalid unresolved type up reference");
  
  return false;
}

/// HandleUpRefs - Every time we finish a new layer of types, this function is
/// called.  It loops through the UpRefs vector, which is a list of the
/// currently active types.  For each type, if the up-reference is contained in
/// the newly completed type, we decrement the level count.  When the level
/// count reaches zero, the up-referenced type is the type that is passed in:
/// thus we can complete the cycle.
///
PATypeHolder LLParser::HandleUpRefs(const Type *ty) {
  // If Ty isn't abstract, or if there are no up-references in it, then there is
  // nothing to resolve here.
  if (!ty->isAbstract() || UpRefs.empty()) return ty;
  
  PATypeHolder Ty(ty);
#if 0
  errs() << "Type '" << Ty->getDescription()
         << "' newly formed.  Resolving upreferences.\n"
         << UpRefs.size() << " upreferences active!\n";
#endif
  
  // If we find any resolvable upreferences (i.e., those whose NestingLevel goes
  // to zero), we resolve them all together before we resolve them to Ty.  At
  // the end of the loop, if there is anything to resolve to Ty, it will be in
  // this variable.
  OpaqueType *TypeToResolve = 0;
  
  for (unsigned i = 0; i != UpRefs.size(); ++i) {
    // Determine if 'Ty' directly contains this up-references 'LastContainedTy'.
    bool ContainsType =
      std::find(Ty->subtype_begin(), Ty->subtype_end(),
                UpRefs[i].LastContainedTy) != Ty->subtype_end();
    
#if 0
    errs() << "  UR#" << i << " - TypeContains(" << Ty->getDescription() << ", "
           << UpRefs[i].LastContainedTy->getDescription() << ") = "
           << (ContainsType ? "true" : "false")
           << " level=" << UpRefs[i].NestingLevel << "\n";
#endif
    if (!ContainsType)
      continue;
    
    // Decrement level of upreference
    unsigned Level = --UpRefs[i].NestingLevel;
    UpRefs[i].LastContainedTy = Ty;
    
    // If the Up-reference has a non-zero level, it shouldn't be resolved yet.
    if (Level != 0)
      continue;
    
#if 0
    errs() << "  * Resolving upreference for " << UpRefs[i].UpRefTy << "\n";
#endif
    if (!TypeToResolve)
      TypeToResolve = UpRefs[i].UpRefTy;
    else
      UpRefs[i].UpRefTy->refineAbstractTypeTo(TypeToResolve);
    UpRefs.erase(UpRefs.begin()+i);     // Remove from upreference list.
    --i;                                // Do not skip the next element.
  }
  
  if (TypeToResolve)
    TypeToResolve->refineAbstractTypeTo(Ty);
  
  return Ty;
}


/// ParseTypeRec - The recursive function used to process the internal
/// implementation details of types.
bool LLParser::ParseTypeRec(PATypeHolder &Result) {
  switch (Lex.getKind()) {
  default:
    return TokError("expected type");
  case lltok::Type:
    // TypeRec ::= 'float' | 'void' (etc)
    Result = Lex.getTyVal();
    Lex.Lex(); 
    break;
  case lltok::kw_opaque:
    // TypeRec ::= 'opaque'
    Result = OpaqueType::get();
    Lex.Lex();
    break;
  case lltok::lbrace:
    // TypeRec ::= '{' ... '}'
    if (ParseStructType(Result, false))
      return true;
    break;
  case lltok::lsquare:
    // TypeRec ::= '[' ... ']'
    Lex.Lex(); // eat the lsquare.
    if (ParseArrayVectorType(Result, false))
      return true;
    break;
  case lltok::less: // Either vector or packed struct.
    // TypeRec ::= '<' ... '>'
    Lex.Lex();
    if (Lex.getKind() == lltok::lbrace) {
      if (ParseStructType(Result, true) ||
          ParseToken(lltok::greater, "expected '>' at end of packed struct"))
        return true;
    } else if (ParseArrayVectorType(Result, true))
      return true;
    break;
  case lltok::LocalVar:
  case lltok::StringConstant:  // FIXME: REMOVE IN LLVM 3.0
    // TypeRec ::= %foo
    if (const Type *T = M->getTypeByName(Lex.getStrVal())) {
      Result = T;
    } else {
      Result = OpaqueType::get();
      ForwardRefTypes.insert(std::make_pair(Lex.getStrVal(),
                                            std::make_pair(Result,
                                                           Lex.getLoc())));
      M->addTypeName(Lex.getStrVal(), Result.get());
    }
    Lex.Lex();
    break;
      
  case lltok::LocalVarID:
    // TypeRec ::= %4
    if (Lex.getUIntVal() < NumberedTypes.size())
      Result = NumberedTypes[Lex.getUIntVal()];
    else {
      std::map<unsigned, std::pair<PATypeHolder, LocTy> >::iterator
        I = ForwardRefTypeIDs.find(Lex.getUIntVal());
      if (I != ForwardRefTypeIDs.end())
        Result = I->second.first;
      else {
        Result = OpaqueType::get();
        ForwardRefTypeIDs.insert(std::make_pair(Lex.getUIntVal(),
                                                std::make_pair(Result,
                                                               Lex.getLoc())));
      }
    }
    Lex.Lex();
    break;
  case lltok::backslash: {
    // TypeRec ::= '\' 4
    Lex.Lex();
    unsigned Val;
    if (ParseUInt32(Val)) return true;
    OpaqueType *OT = OpaqueType::get();        // Use temporary placeholder.
    UpRefs.push_back(UpRefRecord(Lex.getLoc(), Val, OT));
    Result = OT;
    break;
  }
  }
  
  // Parse the type suffixes. 
  while (1) {
    switch (Lex.getKind()) {
    // End of type.
    default: return false;    

    // TypeRec ::= TypeRec '*'
    case lltok::star:
      if (Result.get() == Type::LabelTy)
        return TokError("basic block pointers are invalid");
      Result = HandleUpRefs(PointerType::getUnqual(Result.get()));
      Lex.Lex();
      break;

    // TypeRec ::= TypeRec 'addrspace' '(' uint32 ')' '*'
    case lltok::kw_addrspace: {
      if (Result.get() == Type::LabelTy)
        return TokError("basic block pointers are invalid");
      unsigned AddrSpace;
      if (ParseOptionalAddrSpace(AddrSpace) ||
          ParseToken(lltok::star, "expected '*' in address space"))
        return true;

      Result = HandleUpRefs(PointerType::get(Result.get(), AddrSpace));
      break;
    }
        
    /// Types '(' ArgTypeListI ')' OptFuncAttrs
    case lltok::lparen:
      if (ParseFunctionType(Result))
        return true;
      break;
    }
  }
}

/// ParseParameterList
///    ::= '(' ')'
///    ::= '(' Arg (',' Arg)* ')'
///  Arg
///    ::= Type OptionalAttributes Value OptionalAttributes
bool LLParser::ParseParameterList(SmallVectorImpl<ParamInfo> &ArgList,
                                  PerFunctionState &PFS) {
  if (ParseToken(lltok::lparen, "expected '(' in call"))
    return true;
  
  while (Lex.getKind() != lltok::rparen) {
    // If this isn't the first argument, we need a comma.
    if (!ArgList.empty() &&
        ParseToken(lltok::comma, "expected ',' in argument list"))
      return true;
    
    // Parse the argument.
    LocTy ArgLoc;
    PATypeHolder ArgTy(Type::VoidTy);
    unsigned ArgAttrs1, ArgAttrs2;
    Value *V;
    if (ParseType(ArgTy, ArgLoc) ||
        ParseOptionalAttrs(ArgAttrs1, 0) ||
        ParseValue(ArgTy, V, PFS) ||
        // FIXME: Should not allow attributes after the argument, remove this in
        // LLVM 3.0.
        ParseOptionalAttrs(ArgAttrs2, 0))
      return true;
    ArgList.push_back(ParamInfo(ArgLoc, V, ArgAttrs1|ArgAttrs2));
  }

  Lex.Lex();  // Lex the ')'.
  return false;
}



/// ParseArgumentList - Parse the argument list for a function type or function
/// prototype.  If 'inType' is true then we are parsing a FunctionType.
///   ::= '(' ArgTypeListI ')'
/// ArgTypeListI
///   ::= /*empty*/
///   ::= '...'
///   ::= ArgTypeList ',' '...'
///   ::= ArgType (',' ArgType)*
///
bool LLParser::ParseArgumentList(std::vector<ArgInfo> &ArgList,
                                 bool &isVarArg, bool inType) {
  isVarArg = false;
  assert(Lex.getKind() == lltok::lparen);
  Lex.Lex(); // eat the (.
  
  if (Lex.getKind() == lltok::rparen) {
    // empty
  } else if (Lex.getKind() == lltok::dotdotdot) {
    isVarArg = true;
    Lex.Lex();
  } else {
    LocTy TypeLoc = Lex.getLoc();
    PATypeHolder ArgTy(Type::VoidTy);
    unsigned Attrs;
    std::string Name;
    
    // If we're parsing a type, use ParseTypeRec, because we allow recursive
    // types (such as a function returning a pointer to itself).  If parsing a
    // function prototype, we require fully resolved types.
    if ((inType ? ParseTypeRec(ArgTy) : ParseType(ArgTy)) ||
        ParseOptionalAttrs(Attrs, 0)) return true;
    
    if (Lex.getKind() == lltok::LocalVar ||
        Lex.getKind() == lltok::StringConstant) { // FIXME: REMOVE IN LLVM 3.0
      Name = Lex.getStrVal();
      Lex.Lex();
    }

    if (!ArgTy->isFirstClassType() && !isa<OpaqueType>(ArgTy))
      return Error(TypeLoc, "invalid type for function argument");
    
    ArgList.push_back(ArgInfo(TypeLoc, ArgTy, Attrs, Name));
    
    while (EatIfPresent(lltok::comma)) {
      // Handle ... at end of arg list.
      if (EatIfPresent(lltok::dotdotdot)) {
        isVarArg = true;
        break;
      }
      
      // Otherwise must be an argument type.
      TypeLoc = Lex.getLoc();
      if (ParseTypeRec(ArgTy) ||
          ParseOptionalAttrs(Attrs, 0)) return true;

      if (Lex.getKind() == lltok::LocalVar ||
          Lex.getKind() == lltok::StringConstant) { // FIXME: REMOVE IN LLVM 3.0
        Name = Lex.getStrVal();
        Lex.Lex();
      } else {
        Name = "";
      }

      if (!ArgTy->isFirstClassType() && !isa<OpaqueType>(ArgTy))
        return Error(TypeLoc, "invalid type for function argument");
      
      ArgList.push_back(ArgInfo(TypeLoc, ArgTy, Attrs, Name));
    }
  }
  
  return ParseToken(lltok::rparen, "expected ')' at end of argument list");
}
  
/// ParseFunctionType
///  ::= Type ArgumentList OptionalAttrs
bool LLParser::ParseFunctionType(PATypeHolder &Result) {
  assert(Lex.getKind() == lltok::lparen);

  if (!FunctionType::isValidReturnType(Result))
    return TokError("invalid function return type");
  
  std::vector<ArgInfo> ArgList;
  bool isVarArg;
  unsigned Attrs;
  if (ParseArgumentList(ArgList, isVarArg, true) ||
      // FIXME: Allow, but ignore attributes on function types!
      // FIXME: Remove in LLVM 3.0
      ParseOptionalAttrs(Attrs, 2))
    return true;
  
  // Reject names on the arguments lists.
  for (unsigned i = 0, e = ArgList.size(); i != e; ++i) {
    if (!ArgList[i].Name.empty())
      return Error(ArgList[i].Loc, "argument name invalid in function type");
    if (!ArgList[i].Attrs != 0) {
      // Allow but ignore attributes on function types; this permits
      // auto-upgrade.
      // FIXME: REJECT ATTRIBUTES ON FUNCTION TYPES in LLVM 3.0
    }
  }
  
  std::vector<const Type*> ArgListTy;
  for (unsigned i = 0, e = ArgList.size(); i != e; ++i)
    ArgListTy.push_back(ArgList[i].Type);
    
  Result = HandleUpRefs(FunctionType::get(Result.get(), ArgListTy, isVarArg));
  return false;
}

/// ParseStructType: Handles packed and unpacked types.  </> parsed elsewhere.
///   TypeRec
///     ::= '{' '}'
///     ::= '{' TypeRec (',' TypeRec)* '}'
///     ::= '<' '{' '}' '>'
///     ::= '<' '{' TypeRec (',' TypeRec)* '}' '>'
bool LLParser::ParseStructType(PATypeHolder &Result, bool Packed) {
  assert(Lex.getKind() == lltok::lbrace);
  Lex.Lex(); // Consume the '{'
  
  if (EatIfPresent(lltok::rbrace)) {
    Result = StructType::get(std::vector<const Type*>(), Packed);
    return false;
  }

  std::vector<PATypeHolder> ParamsList;
  if (ParseTypeRec(Result)) return true;
  ParamsList.push_back(Result);
  
  while (EatIfPresent(lltok::comma)) {
    if (ParseTypeRec(Result)) return true;
    ParamsList.push_back(Result);
  }
  
  if (ParseToken(lltok::rbrace, "expected '}' at end of struct"))
    return true;
  
  std::vector<const Type*> ParamsListTy;
  for (unsigned i = 0, e = ParamsList.size(); i != e; ++i)
    ParamsListTy.push_back(ParamsList[i].get());
  Result = HandleUpRefs(StructType::get(ParamsListTy, Packed));
  return false;
}

/// ParseArrayVectorType - Parse an array or vector type, assuming the first
/// token has already been consumed.
///   TypeRec 
///     ::= '[' APSINTVAL 'x' Types ']'
///     ::= '<' APSINTVAL 'x' Types '>'
bool LLParser::ParseArrayVectorType(PATypeHolder &Result, bool isVector) {
  if (Lex.getKind() != lltok::APSInt || Lex.getAPSIntVal().isSigned() ||
      Lex.getAPSIntVal().getBitWidth() > 64)
    return TokError("expected number in address space");
  
  LocTy SizeLoc = Lex.getLoc();
  uint64_t Size = Lex.getAPSIntVal().getZExtValue();
  Lex.Lex();
      
  if (ParseToken(lltok::kw_x, "expected 'x' after element count"))
      return true;

  LocTy TypeLoc = Lex.getLoc();
  PATypeHolder EltTy(Type::VoidTy);
  if (ParseTypeRec(EltTy)) return true;
  
  if (ParseToken(isVector ? lltok::greater : lltok::rsquare,
                 "expected end of sequential type"))
    return true;
  
  if (isVector) {
    if ((unsigned)Size != Size)
      return Error(SizeLoc, "size too large for vector");
    if (!EltTy->isFloatingPoint() && !EltTy->isInteger())
      return Error(TypeLoc, "vector element type must be fp or integer");
    Result = VectorType::get(EltTy, unsigned(Size));
  } else {
    if (!EltTy->isFirstClassType() && !isa<OpaqueType>(EltTy))
      return Error(TypeLoc, "invalid array element type");
    Result = HandleUpRefs(ArrayType::get(EltTy, Size));
  }
  return false;
}

//===----------------------------------------------------------------------===//
// Function Semantic Analysis.
//===----------------------------------------------------------------------===//

LLParser::PerFunctionState::PerFunctionState(LLParser &p, Function &f)
  : P(p), F(f) {

  // Insert unnamed arguments into the NumberedVals list.
  for (Function::arg_iterator AI = F.arg_begin(), E = F.arg_end();
       AI != E; ++AI)
    if (!AI->hasName())
      NumberedVals.push_back(AI);
}

LLParser::PerFunctionState::~PerFunctionState() {
  // If there were any forward referenced non-basicblock values, delete them.
  for (std::map<std::string, std::pair<Value*, LocTy> >::iterator
       I = ForwardRefVals.begin(), E = ForwardRefVals.end(); I != E; ++I)
    if (!isa<BasicBlock>(I->second.first)) {
      I->second.first->replaceAllUsesWith(UndefValue::get(I->second.first
                                                          ->getType()));
      delete I->second.first;
      I->second.first = 0;
    }
  
  for (std::map<unsigned, std::pair<Value*, LocTy> >::iterator
       I = ForwardRefValIDs.begin(), E = ForwardRefValIDs.end(); I != E; ++I)
    if (!isa<BasicBlock>(I->second.first)) {
      I->second.first->replaceAllUsesWith(UndefValue::get(I->second.first
                                                          ->getType()));
      delete I->second.first;
      I->second.first = 0;
    }
}

bool LLParser::PerFunctionState::VerifyFunctionComplete() {
  if (!ForwardRefVals.empty())
    return P.Error(ForwardRefVals.begin()->second.second,
                   "use of undefined value '%" + ForwardRefVals.begin()->first +
                   "'");
  if (!ForwardRefValIDs.empty())
    return P.Error(ForwardRefValIDs.begin()->second.second,
                   "use of undefined value '%" +
                   utostr(ForwardRefValIDs.begin()->first) + "'");
  return false;
}


/// GetVal - Get a value with the specified name or ID, creating a
/// forward reference record if needed.  This can return null if the value
/// exists but does not have the right type.
Value *LLParser::PerFunctionState::GetVal(const std::string &Name,
                                          const Type *Ty, LocTy Loc) {
  // Look this name up in the normal function symbol table.
  Value *Val = F.getValueSymbolTable().lookup(Name);
  
  // If this is a forward reference for the value, see if we already created a
  // forward ref record.
  if (Val == 0) {
    std::map<std::string, std::pair<Value*, LocTy> >::iterator
      I = ForwardRefVals.find(Name);
    if (I != ForwardRefVals.end())
      Val = I->second.first;
  }
    
  // If we have the value in the symbol table or fwd-ref table, return it.
  if (Val) {
    if (Val->getType() == Ty) return Val;
    if (Ty == Type::LabelTy)
      P.Error(Loc, "'%" + Name + "' is not a basic block");
    else
      P.Error(Loc, "'%" + Name + "' defined with type '" +
              Val->getType()->getDescription() + "'");
    return 0;
  }
  
  // Don't make placeholders with invalid type.
  if (!Ty->isFirstClassType() && !isa<OpaqueType>(Ty) && Ty != Type::LabelTy) {
    P.Error(Loc, "invalid use of a non-first-class type");
    return 0;
  }
  
  // Otherwise, create a new forward reference for this value and remember it.
  Value *FwdVal;
  if (Ty == Type::LabelTy) 
    FwdVal = BasicBlock::Create(Name, &F);
  else
    FwdVal = new Argument(Ty, Name);
  
  ForwardRefVals[Name] = std::make_pair(FwdVal, Loc);
  return FwdVal;
}

Value *LLParser::PerFunctionState::GetVal(unsigned ID, const Type *Ty,
                                          LocTy Loc) {
  // Look this name up in the normal function symbol table.
  Value *Val = ID < NumberedVals.size() ? NumberedVals[ID] : 0;
  
  // If this is a forward reference for the value, see if we already created a
  // forward ref record.
  if (Val == 0) {
    std::map<unsigned, std::pair<Value*, LocTy> >::iterator
      I = ForwardRefValIDs.find(ID);
    if (I != ForwardRefValIDs.end())
      Val = I->second.first;
  }
  
  // If we have the value in the symbol table or fwd-ref table, return it.
  if (Val) {
    if (Val->getType() == Ty) return Val;
    if (Ty == Type::LabelTy)
      P.Error(Loc, "'%" + utostr(ID) + "' is not a basic block");
    else
      P.Error(Loc, "'%" + utostr(ID) + "' defined with type '" +
              Val->getType()->getDescription() + "'");
    return 0;
  }
  
  if (!Ty->isFirstClassType() && !isa<OpaqueType>(Ty) && Ty != Type::LabelTy) {
    P.Error(Loc, "invalid use of a non-first-class type");
    return 0;
  }
  
  // Otherwise, create a new forward reference for this value and remember it.
  Value *FwdVal;
  if (Ty == Type::LabelTy) 
    FwdVal = BasicBlock::Create("", &F);
  else
    FwdVal = new Argument(Ty);
  
  ForwardRefValIDs[ID] = std::make_pair(FwdVal, Loc);
  return FwdVal;
}

/// SetInstName - After an instruction is parsed and inserted into its
/// basic block, this installs its name.
bool LLParser::PerFunctionState::SetInstName(int NameID,
                                             const std::string &NameStr,
                                             LocTy NameLoc, Instruction *Inst) {
  // If this instruction has void type, it cannot have a name or ID specified.
  if (Inst->getType() == Type::VoidTy) {
    if (NameID != -1 || !NameStr.empty())
      return P.Error(NameLoc, "instructions returning void cannot have a name");
    return false;
  }
  
  // If this was a numbered instruction, verify that the instruction is the
  // expected value and resolve any forward references.
  if (NameStr.empty()) {
    // If neither a name nor an ID was specified, just use the next ID.
    if (NameID == -1)
      NameID = NumberedVals.size();
    
    if (unsigned(NameID) != NumberedVals.size())
      return P.Error(NameLoc, "instruction expected to be numbered '%" +
                     utostr(NumberedVals.size()) + "'");
    
    std::map<unsigned, std::pair<Value*, LocTy> >::iterator FI =
      ForwardRefValIDs.find(NameID);
    if (FI != ForwardRefValIDs.end()) {
      if (FI->second.first->getType() != Inst->getType())
        return P.Error(NameLoc, "instruction forward referenced with type '" + 
                       FI->second.first->getType()->getDescription() + "'");
      FI->second.first->replaceAllUsesWith(Inst);
      ForwardRefValIDs.erase(FI);
    }

    NumberedVals.push_back(Inst);
    return false;
  }

  // Otherwise, the instruction had a name.  Resolve forward refs and set it.
  std::map<std::string, std::pair<Value*, LocTy> >::iterator
    FI = ForwardRefVals.find(NameStr);
  if (FI != ForwardRefVals.end()) {
    if (FI->second.first->getType() != Inst->getType())
      return P.Error(NameLoc, "instruction forward referenced with type '" + 
                     FI->second.first->getType()->getDescription() + "'");
    FI->second.first->replaceAllUsesWith(Inst);
    ForwardRefVals.erase(FI);
  }
  
  // Set the name on the instruction.
  Inst->setName(NameStr);
  
  if (Inst->getNameStr() != NameStr)
    return P.Error(NameLoc, "multiple definition of local value named '" + 
                   NameStr + "'");
  return false;
}

/// GetBB - Get a basic block with the specified name or ID, creating a
/// forward reference record if needed.
BasicBlock *LLParser::PerFunctionState::GetBB(const std::string &Name,
                                              LocTy Loc) {
  return cast_or_null<BasicBlock>(GetVal(Name, Type::LabelTy, Loc));
}

BasicBlock *LLParser::PerFunctionState::GetBB(unsigned ID, LocTy Loc) {
  return cast_or_null<BasicBlock>(GetVal(ID, Type::LabelTy, Loc));
}

/// DefineBB - Define the specified basic block, which is either named or
/// unnamed.  If there is an error, this returns null otherwise it returns
/// the block being defined.
BasicBlock *LLParser::PerFunctionState::DefineBB(const std::string &Name,
                                                 LocTy Loc) {
  BasicBlock *BB;
  if (Name.empty())
    BB = GetBB(NumberedVals.size(), Loc);
  else
    BB = GetBB(Name, Loc);
  if (BB == 0) return 0; // Already diagnosed error.
  
  // Move the block to the end of the function.  Forward ref'd blocks are
  // inserted wherever they happen to be referenced.
  F.getBasicBlockList().splice(F.end(), F.getBasicBlockList(), BB);
  
  // Remove the block from forward ref sets.
  if (Name.empty()) {
    ForwardRefValIDs.erase(NumberedVals.size());
    NumberedVals.push_back(BB);
  } else {
    // BB forward references are already in the function symbol table.
    ForwardRefVals.erase(Name);
  }
  
  return BB;
}

//===----------------------------------------------------------------------===//
// Constants.
//===----------------------------------------------------------------------===//

/// ParseValID - Parse an abstract value that doesn't necessarily have a
/// type implied.  For example, if we parse "4" we don't know what integer type
/// it has.  The value will later be combined with its type and checked for
/// sanity.
bool LLParser::ParseValID(ValID &ID) {
  ID.Loc = Lex.getLoc();
  switch (Lex.getKind()) {
  default: return TokError("expected value token");
  case lltok::GlobalID:  // @42
    ID.UIntVal = Lex.getUIntVal();
    ID.Kind = ValID::t_GlobalID;
    break;
  case lltok::GlobalVar:  // @foo
    ID.StrVal = Lex.getStrVal();
    ID.Kind = ValID::t_GlobalName;
    break;
  case lltok::LocalVarID:  // %42
    ID.UIntVal = Lex.getUIntVal();
    ID.Kind = ValID::t_LocalID;
    break;
  case lltok::LocalVar:  // %foo
  case lltok::StringConstant:  // "foo" - FIXME: REMOVE IN LLVM 3.0
    ID.StrVal = Lex.getStrVal();
    ID.Kind = ValID::t_LocalName;
    break;
  case lltok::APSInt:
    ID.APSIntVal = Lex.getAPSIntVal(); 
    ID.Kind = ValID::t_APSInt;
    break;
  case lltok::APFloat:
    ID.APFloatVal = Lex.getAPFloatVal();
    ID.Kind = ValID::t_APFloat;
    break;
  case lltok::kw_true:
    ID.ConstantVal = ConstantInt::getTrue();
    ID.Kind = ValID::t_Constant;
    break;
  case lltok::kw_false:
    ID.ConstantVal = ConstantInt::getFalse();
    ID.Kind = ValID::t_Constant;
    break;
  case lltok::kw_null: ID.Kind = ValID::t_Null; break;
  case lltok::kw_undef: ID.Kind = ValID::t_Undef; break;
  case lltok::kw_zeroinitializer: ID.Kind = ValID::t_Zero; break;
      
  case lltok::lbrace: {
    // ValID ::= '{' ConstVector '}'
    Lex.Lex();
    SmallVector<Constant*, 16> Elts;
    if (ParseGlobalValueVector(Elts) ||
        ParseToken(lltok::rbrace, "expected end of struct constant"))
      return true;
    
    ID.ConstantVal = ConstantStruct::get(&Elts[0], Elts.size(), false);
    ID.Kind = ValID::t_Constant;
    return false;
  }
  case lltok::less: {
    // ValID ::= '<' ConstVector '>'         --> Vector.
    // ValID ::= '<' '{' ConstVector '}' '>' --> Packed Struct.
    Lex.Lex();
    bool isPackedStruct = EatIfPresent(lltok::lbrace);
    
    SmallVector<Constant*, 16> Elts;
    LocTy FirstEltLoc = Lex.getLoc();
    if (ParseGlobalValueVector(Elts) ||
        (isPackedStruct &&
         ParseToken(lltok::rbrace, "expected end of packed struct")) ||
        ParseToken(lltok::greater, "expected end of constant"))
      return true;
    
    if (isPackedStruct) {
      ID.ConstantVal = ConstantStruct::get(&Elts[0], Elts.size(), true);
      ID.Kind = ValID::t_Constant;
      return false;
    }
    
    if (Elts.empty())
      return Error(ID.Loc, "constant vector must not be empty");

    if (!Elts[0]->getType()->isInteger() &&
        !Elts[0]->getType()->isFloatingPoint())
      return Error(FirstEltLoc,
                   "vector elements must have integer or floating point type");
    
    // Verify that all the vector elements have the same type.
    for (unsigned i = 1, e = Elts.size(); i != e; ++i)
      if (Elts[i]->getType() != Elts[0]->getType())
        return Error(FirstEltLoc,
                     "vector element #" + utostr(i) +
                    " is not of type '" + Elts[0]->getType()->getDescription());
    
    ID.ConstantVal = ConstantVector::get(&Elts[0], Elts.size());
    ID.Kind = ValID::t_Constant;
    return false;
  }
  case lltok::lsquare: {   // Array Constant
    Lex.Lex();
    SmallVector<Constant*, 16> Elts;
    LocTy FirstEltLoc = Lex.getLoc();
    if (ParseGlobalValueVector(Elts) ||
        ParseToken(lltok::rsquare, "expected end of array constant"))
      return true;

    // Handle empty element.
    if (Elts.empty()) {
      // Use undef instead of an array because it's inconvenient to determine
      // the element type at this point, there being no elements to examine.
      ID.Kind = ValID::t_EmptyArray;
      return false;
    }
    
    if (!Elts[0]->getType()->isFirstClassType())
      return Error(FirstEltLoc, "invalid array element type: " + 
                   Elts[0]->getType()->getDescription());
          
    ArrayType *ATy = ArrayType::get(Elts[0]->getType(), Elts.size());
    
    // Verify all elements are correct type!
    for (unsigned i = 0, e = Elts.size(); i != e; ++i) {
      if (Elts[i]->getType() != Elts[0]->getType())
        return Error(FirstEltLoc,
                     "array element #" + utostr(i) +
                     " is not of type '" +Elts[0]->getType()->getDescription());
    }
          
    ID.ConstantVal = ConstantArray::get(ATy, &Elts[0], Elts.size());
    ID.Kind = ValID::t_Constant;
    return false;
  }
  case lltok::kw_c:  // c "foo"
    Lex.Lex();
    ID.ConstantVal = ConstantArray::get(Lex.getStrVal(), false);
    if (ParseToken(lltok::StringConstant, "expected string")) return true;
    ID.Kind = ValID::t_Constant;
    return false;

  case lltok::kw_asm: {
    // ValID ::= 'asm' SideEffect? STRINGCONSTANT ',' STRINGCONSTANT
    bool HasSideEffect;
    Lex.Lex();
    if (ParseOptionalToken(lltok::kw_sideeffect, HasSideEffect) ||
        ParseStringConstant(ID.StrVal) ||
        ParseToken(lltok::comma, "expected comma in inline asm expression") ||
        ParseToken(lltok::StringConstant, "expected constraint string"))
      return true;
    ID.StrVal2 = Lex.getStrVal();
    ID.UIntVal = HasSideEffect;
    ID.Kind = ValID::t_InlineAsm;
    return false;
  }
      
  case lltok::kw_trunc:
  case lltok::kw_zext:
  case lltok::kw_sext:
  case lltok::kw_fptrunc:
  case lltok::kw_fpext:
  case lltok::kw_bitcast:
  case lltok::kw_uitofp:
  case lltok::kw_sitofp:
  case lltok::kw_fptoui:
  case lltok::kw_fptosi: 
  case lltok::kw_inttoptr:
  case lltok::kw_ptrtoint: { 
    unsigned Opc = Lex.getUIntVal();
    PATypeHolder DestTy(Type::VoidTy);
    Constant *SrcVal;
    Lex.Lex();
    if (ParseToken(lltok::lparen, "expected '(' after constantexpr cast") ||
        ParseGlobalTypeAndValue(SrcVal) ||
        ParseToken(lltok::kw_to, "expected 'to' int constantexpr cast") ||
        ParseType(DestTy) ||
        ParseToken(lltok::rparen, "expected ')' at end of constantexpr cast"))
      return true;
    if (!CastInst::castIsValid((Instruction::CastOps)Opc, SrcVal, DestTy))
      return Error(ID.Loc, "invalid cast opcode for cast from '" +
                   SrcVal->getType()->getDescription() + "' to '" +
                   DestTy->getDescription() + "'");
    ID.ConstantVal = ConstantExpr::getCast((Instruction::CastOps)Opc, SrcVal,
                                           DestTy);
    ID.Kind = ValID::t_Constant;
    return false;
  }
  case lltok::kw_extractvalue: {
    Lex.Lex();
    Constant *Val;
    SmallVector<unsigned, 4> Indices;
    if (ParseToken(lltok::lparen, "expected '(' in extractvalue constantexpr")||
        ParseGlobalTypeAndValue(Val) ||
        ParseIndexList(Indices) ||
        ParseToken(lltok::rparen, "expected ')' in extractvalue constantexpr"))
      return true;
    if (!isa<StructType>(Val->getType()) && !isa<ArrayType>(Val->getType()))
      return Error(ID.Loc, "extractvalue operand must be array or struct");
    if (!ExtractValueInst::getIndexedType(Val->getType(), Indices.begin(),
                                          Indices.end()))
      return Error(ID.Loc, "invalid indices for extractvalue");
    ID.ConstantVal = ConstantExpr::getExtractValue(Val,
                                                   &Indices[0], Indices.size());
    ID.Kind = ValID::t_Constant;
    return false;
  }
  case lltok::kw_insertvalue: {
    Lex.Lex();
    Constant *Val0, *Val1;
    SmallVector<unsigned, 4> Indices;
    if (ParseToken(lltok::lparen, "expected '(' in insertvalue constantexpr")||
        ParseGlobalTypeAndValue(Val0) ||
        ParseToken(lltok::comma, "expected comma in insertvalue constantexpr")||
        ParseGlobalTypeAndValue(Val1) ||
        ParseIndexList(Indices) ||
        ParseToken(lltok::rparen, "expected ')' in insertvalue constantexpr"))
      return true;
    if (!isa<StructType>(Val0->getType()) && !isa<ArrayType>(Val0->getType()))
      return Error(ID.Loc, "extractvalue operand must be array or struct");
    if (!ExtractValueInst::getIndexedType(Val0->getType(), Indices.begin(),
                                          Indices.end()))
      return Error(ID.Loc, "invalid indices for insertvalue");
    ID.ConstantVal = ConstantExpr::getInsertValue(Val0, Val1,
                                                  &Indices[0], Indices.size());
    ID.Kind = ValID::t_Constant;
    return false;
  }
  case lltok::kw_icmp:
  case lltok::kw_fcmp:
  case lltok::kw_vicmp:
  case lltok::kw_vfcmp: {
    unsigned PredVal, Opc = Lex.getUIntVal();
    Constant *Val0, *Val1;
    Lex.Lex();
    if (ParseCmpPredicate(PredVal, Opc) ||
        ParseToken(lltok::lparen, "expected '(' in compare constantexpr") ||
        ParseGlobalTypeAndValue(Val0) ||
        ParseToken(lltok::comma, "expected comma in compare constantexpr") ||
        ParseGlobalTypeAndValue(Val1) ||
        ParseToken(lltok::rparen, "expected ')' in compare constantexpr"))
      return true;
    
    if (Val0->getType() != Val1->getType())
      return Error(ID.Loc, "compare operands must have the same type");
    
    CmpInst::Predicate Pred = (CmpInst::Predicate)PredVal;
    
    if (Opc == Instruction::FCmp) {
      if (!Val0->getType()->isFPOrFPVector())
        return Error(ID.Loc, "fcmp requires floating point operands");
      ID.ConstantVal = ConstantExpr::getFCmp(Pred, Val0, Val1);
    } else if (Opc == Instruction::ICmp) {
      if (!Val0->getType()->isIntOrIntVector() &&
          !isa<PointerType>(Val0->getType()))
        return Error(ID.Loc, "icmp requires pointer or integer operands");
      ID.ConstantVal = ConstantExpr::getICmp(Pred, Val0, Val1);
    } else if (Opc == Instruction::VFCmp) {
      // FIXME: REMOVE VFCMP Support
      if (!Val0->getType()->isFPOrFPVector() ||
          !isa<VectorType>(Val0->getType()))
        return Error(ID.Loc, "vfcmp requires vector floating point operands");
      ID.ConstantVal = ConstantExpr::getVFCmp(Pred, Val0, Val1);
    } else if (Opc == Instruction::VICmp) {
      // FIXME: REMOVE VICMP Support
      if (!Val0->getType()->isIntOrIntVector() ||
          !isa<VectorType>(Val0->getType()))
        return Error(ID.Loc, "vicmp requires vector floating point operands");
      ID.ConstantVal = ConstantExpr::getVICmp(Pred, Val0, Val1);
    }
    ID.Kind = ValID::t_Constant;
    return false;
  }
      
  // Binary Operators.
  case lltok::kw_add:
  case lltok::kw_sub:
  case lltok::kw_mul:
  case lltok::kw_udiv:
  case lltok::kw_sdiv:
  case lltok::kw_fdiv:
  case lltok::kw_urem:
  case lltok::kw_srem:
  case lltok::kw_frem: {
    unsigned Opc = Lex.getUIntVal();
    Constant *Val0, *Val1;
    Lex.Lex();
    if (ParseToken(lltok::lparen, "expected '(' in binary constantexpr") ||
        ParseGlobalTypeAndValue(Val0) ||
        ParseToken(lltok::comma, "expected comma in binary constantexpr") ||
        ParseGlobalTypeAndValue(Val1) ||
        ParseToken(lltok::rparen, "expected ')' in binary constantexpr"))
      return true;
    if (Val0->getType() != Val1->getType())
      return Error(ID.Loc, "operands of constexpr must have same type");
    if (!Val0->getType()->isIntOrIntVector() &&
        !Val0->getType()->isFPOrFPVector())
      return Error(ID.Loc,"constexpr requires integer, fp, or vector operands");
    ID.ConstantVal = ConstantExpr::get(Opc, Val0, Val1);
    ID.Kind = ValID::t_Constant;
    return false;
  }
      
  // Logical Operations
  case lltok::kw_shl:
  case lltok::kw_lshr:
  case lltok::kw_ashr:
  case lltok::kw_and:
  case lltok::kw_or:
  case lltok::kw_xor: {
    unsigned Opc = Lex.getUIntVal();
    Constant *Val0, *Val1;
    Lex.Lex();
    if (ParseToken(lltok::lparen, "expected '(' in logical constantexpr") ||
        ParseGlobalTypeAndValue(Val0) ||
        ParseToken(lltok::comma, "expected comma in logical constantexpr") ||
        ParseGlobalTypeAndValue(Val1) ||
        ParseToken(lltok::rparen, "expected ')' in logical constantexpr"))
      return true;
    if (Val0->getType() != Val1->getType())
      return Error(ID.Loc, "operands of constexpr must have same type");
    if (!Val0->getType()->isIntOrIntVector())
      return Error(ID.Loc,
                   "constexpr requires integer or integer vector operands");
    ID.ConstantVal = ConstantExpr::get(Opc, Val0, Val1);
    ID.Kind = ValID::t_Constant;
    return false;
  }  
      
  case lltok::kw_getelementptr:
  case lltok::kw_shufflevector:
  case lltok::kw_insertelement:
  case lltok::kw_extractelement:
  case lltok::kw_select: {
    unsigned Opc = Lex.getUIntVal();
    SmallVector<Constant*, 16> Elts;
    Lex.Lex();
    if (ParseToken(lltok::lparen, "expected '(' in constantexpr") ||
        ParseGlobalValueVector(Elts) ||
        ParseToken(lltok::rparen, "expected ')' in constantexpr"))
      return true;
    
    if (Opc == Instruction::GetElementPtr) {
      if (Elts.size() == 0 || !isa<PointerType>(Elts[0]->getType()))
        return Error(ID.Loc, "getelementptr requires pointer operand");
      
      if (!GetElementPtrInst::getIndexedType(Elts[0]->getType(),
                                             (Value**)&Elts[1], Elts.size()-1))
        return Error(ID.Loc, "invalid indices for getelementptr");
      ID.ConstantVal = ConstantExpr::getGetElementPtr(Elts[0],
                                                      &Elts[1], Elts.size()-1);
    } else if (Opc == Instruction::Select) {
      if (Elts.size() != 3)
        return Error(ID.Loc, "expected three operands to select");
      if (const char *Reason = SelectInst::areInvalidOperands(Elts[0], Elts[1],
                                                              Elts[2]))
        return Error(ID.Loc, Reason);
      ID.ConstantVal = ConstantExpr::getSelect(Elts[0], Elts[1], Elts[2]);
    } else if (Opc == Instruction::ShuffleVector) {
      if (Elts.size() != 3)
        return Error(ID.Loc, "expected three operands to shufflevector");
      if (!ShuffleVectorInst::isValidOperands(Elts[0], Elts[1], Elts[2]))
        return Error(ID.Loc, "invalid operands to shufflevector");
      ID.ConstantVal = ConstantExpr::getShuffleVector(Elts[0], Elts[1],Elts[2]);
    } else if (Opc == Instruction::ExtractElement) {
      if (Elts.size() != 2)
        return Error(ID.Loc, "expected two operands to extractelement");
      if (!ExtractElementInst::isValidOperands(Elts[0], Elts[1]))
        return Error(ID.Loc, "invalid extractelement operands");
      ID.ConstantVal = ConstantExpr::getExtractElement(Elts[0], Elts[1]);
    } else {
      assert(Opc == Instruction::InsertElement && "Unknown opcode");
      if (Elts.size() != 3)
      return Error(ID.Loc, "expected three operands to insertelement");
      if (!InsertElementInst::isValidOperands(Elts[0], Elts[1], Elts[2]))
        return Error(ID.Loc, "invalid insertelement operands");
      ID.ConstantVal = ConstantExpr::getInsertElement(Elts[0], Elts[1],Elts[2]);
    }
    
    ID.Kind = ValID::t_Constant;
    return false;
  }
  }
  
  Lex.Lex();
  return false;
}

/// ParseGlobalValue - Parse a global value with the specified type.
bool LLParser::ParseGlobalValue(const Type *Ty, Constant *&V) {
  V = 0;
  ValID ID;
  return ParseValID(ID) ||
         ConvertGlobalValIDToValue(Ty, ID, V);
}

/// ConvertGlobalValIDToValue - Apply a type to a ValID to get a fully resolved
/// constant.
bool LLParser::ConvertGlobalValIDToValue(const Type *Ty, ValID &ID,
                                         Constant *&V) {
  if (isa<FunctionType>(Ty))
    return Error(ID.Loc, "functions are not values, refer to them as pointers");
  
  switch (ID.Kind) {
  default: assert(0 && "Unknown ValID!");
  case ValID::t_LocalID:
  case ValID::t_LocalName:
    return Error(ID.Loc, "invalid use of function-local name");
  case ValID::t_InlineAsm:
    return Error(ID.Loc, "inline asm can only be an operand of call/invoke");
  case ValID::t_GlobalName:
    V = GetGlobalVal(ID.StrVal, Ty, ID.Loc);
    return V == 0;
  case ValID::t_GlobalID:
    V = GetGlobalVal(ID.UIntVal, Ty, ID.Loc);
    return V == 0;
  case ValID::t_APSInt:
    if (!isa<IntegerType>(Ty))
      return Error(ID.Loc, "integer constant must have integer type");
    ID.APSIntVal.extOrTrunc(Ty->getPrimitiveSizeInBits());
    V = ConstantInt::get(ID.APSIntVal);
    return false;
  case ValID::t_APFloat:
    if (!Ty->isFloatingPoint() ||
        !ConstantFP::isValueValidForType(Ty, ID.APFloatVal))
      return Error(ID.Loc, "floating point constant invalid for type");
      
    // The lexer has no type info, so builds all float and double FP constants
    // as double.  Fix this here.  Long double does not need this.
    if (&ID.APFloatVal.getSemantics() == &APFloat::IEEEdouble &&
        Ty == Type::FloatTy) {
      bool Ignored;
      ID.APFloatVal.convert(APFloat::IEEEsingle, APFloat::rmNearestTiesToEven,
                            &Ignored);
    }
    V = ConstantFP::get(ID.APFloatVal);
      
    if (V->getType() != Ty)
      return Error(ID.Loc, "floating point constant does not have type '" +
                   Ty->getDescription() + "'");
      
    return false;
  case ValID::t_Null:
    if (!isa<PointerType>(Ty))
      return Error(ID.Loc, "null must be a pointer type");
    V = ConstantPointerNull::get(cast<PointerType>(Ty));
    return false;
  case ValID::t_Undef:
    // FIXME: LabelTy should not be a first-class type.
    if ((!Ty->isFirstClassType() || Ty == Type::LabelTy) &&
        !isa<OpaqueType>(Ty))
      return Error(ID.Loc, "invalid type for undef constant");
    V = UndefValue::get(Ty);
    return false;
  case ValID::t_EmptyArray:
    if (!isa<ArrayType>(Ty) || cast<ArrayType>(Ty)->getNumElements() != 0)
      return Error(ID.Loc, "invalid empty array initializer");
    V = UndefValue::get(Ty);
    return false;
  case ValID::t_Zero:
    // FIXME: LabelTy should not be a first-class type.
    if (!Ty->isFirstClassType() || Ty == Type::LabelTy)
      return Error(ID.Loc, "invalid type for null constant");
    V = Constant::getNullValue(Ty);
    return false;
  case ValID::t_Constant:
    if (ID.ConstantVal->getType() != Ty)
      return Error(ID.Loc, "constant expression type mismatch");
    V = ID.ConstantVal;
    return false;
  }
}
  
bool LLParser::ParseGlobalTypeAndValue(Constant *&V) {
  PATypeHolder Type(Type::VoidTy);
  return ParseType(Type) ||
         ParseGlobalValue(Type, V);
}    

/// ParseGlobalValueVector
///   ::= /*empty*/
///   ::= TypeAndValue (',' TypeAndValue)*
bool LLParser::ParseGlobalValueVector(SmallVectorImpl<Constant*> &Elts) {
  // Empty list.
  if (Lex.getKind() == lltok::rbrace ||
      Lex.getKind() == lltok::rsquare ||
      Lex.getKind() == lltok::greater ||
      Lex.getKind() == lltok::rparen)
    return false;
  
  Constant *C;
  if (ParseGlobalTypeAndValue(C)) return true;
  Elts.push_back(C);
  
  while (EatIfPresent(lltok::comma)) {
    if (ParseGlobalTypeAndValue(C)) return true;
    Elts.push_back(C);
  }
  
  return false;
}


//===----------------------------------------------------------------------===//
// Function Parsing.
//===----------------------------------------------------------------------===//

bool LLParser::ConvertValIDToValue(const Type *Ty, ValID &ID, Value *&V,
                                   PerFunctionState &PFS) {
  if (ID.Kind == ValID::t_LocalID)
    V = PFS.GetVal(ID.UIntVal, Ty, ID.Loc);
  else if (ID.Kind == ValID::t_LocalName)
    V = PFS.GetVal(ID.StrVal, Ty, ID.Loc);
  else if (ID.Kind == ValID::t_InlineAsm) {
    const PointerType *PTy = dyn_cast<PointerType>(Ty);
    const FunctionType *FTy =
      PTy ? dyn_cast<FunctionType>(PTy->getElementType()) : 0;
    if (!FTy || !InlineAsm::Verify(FTy, ID.StrVal2))
      return Error(ID.Loc, "invalid type for inline asm constraint string");
    V = InlineAsm::get(FTy, ID.StrVal, ID.StrVal2, ID.UIntVal);
    return false;
  } else {
    Constant *C;
    if (ConvertGlobalValIDToValue(Ty, ID, C)) return true;
    V = C;
    return false;
  }

  return V == 0;
}

bool LLParser::ParseValue(const Type *Ty, Value *&V, PerFunctionState &PFS) {
  V = 0;
  ValID ID;
  return ParseValID(ID) ||
         ConvertValIDToValue(Ty, ID, V, PFS);
}

bool LLParser::ParseTypeAndValue(Value *&V, PerFunctionState &PFS) {
  PATypeHolder T(Type::VoidTy);
  return ParseType(T) ||
         ParseValue(T, V, PFS);
}

/// FunctionHeader
///   ::= OptionalLinkage OptionalVisibility OptionalCallingConv OptRetAttrs
///       Type GlobalName '(' ArgList ')' OptFuncAttrs OptSection
///       OptionalAlign OptGC
bool LLParser::ParseFunctionHeader(Function *&Fn, bool isDefine) {
  // Parse the linkage.
  LocTy LinkageLoc = Lex.getLoc();
  unsigned Linkage;
  
  unsigned Visibility, CC, RetAttrs;
  PATypeHolder RetType(Type::VoidTy);
  LocTy RetTypeLoc = Lex.getLoc();
  if (ParseOptionalLinkage(Linkage) ||
      ParseOptionalVisibility(Visibility) ||
      ParseOptionalCallingConv(CC) ||
      ParseOptionalAttrs(RetAttrs, 1) ||
      ParseType(RetType, RetTypeLoc))
    return true;

  // Verify that the linkage is ok.
  switch ((GlobalValue::LinkageTypes)Linkage) {
  case GlobalValue::ExternalLinkage:
    break; // always ok.
  case GlobalValue::DLLImportLinkage:
  case GlobalValue::ExternalWeakLinkage:
    if (isDefine)
      return Error(LinkageLoc, "invalid linkage for function definition");
    break;
  case GlobalValue::PrivateLinkage:
  case GlobalValue::InternalLinkage:
  case GlobalValue::LinkOnceLinkage:
  case GlobalValue::WeakLinkage:
  case GlobalValue::DLLExportLinkage:
    if (!isDefine)
      return Error(LinkageLoc, "invalid linkage for function declaration");
    break;
  case GlobalValue::AppendingLinkage:
  case GlobalValue::GhostLinkage:
  case GlobalValue::CommonLinkage:
    return Error(LinkageLoc, "invalid function linkage type");
  }
  
  if (!FunctionType::isValidReturnType(RetType) ||
      isa<OpaqueType>(RetType))
    return Error(RetTypeLoc, "invalid function return type");
  
  if (Lex.getKind() != lltok::GlobalVar)
    return TokError("expected function name");
  
  LocTy NameLoc = Lex.getLoc();
  std::string FunctionName = Lex.getStrVal();
  Lex.Lex();
  
  if (Lex.getKind() != lltok::lparen)
    return TokError("expected '(' in function argument list");
  
  std::vector<ArgInfo> ArgList;
  bool isVarArg;
  unsigned FuncAttrs;
  std::string Section;
  unsigned Alignment;
  std::string GC;

  if (ParseArgumentList(ArgList, isVarArg, false) ||
      ParseOptionalAttrs(FuncAttrs, 2) ||
      (EatIfPresent(lltok::kw_section) &&
       ParseStringConstant(Section)) ||
      ParseOptionalAlignment(Alignment) ||
      (EatIfPresent(lltok::kw_gc) &&
       ParseStringConstant(GC)))
    return true;

  // If the alignment was parsed as an attribute, move to the alignment field.
  if (FuncAttrs & Attribute::Alignment) {
    Alignment = Attribute::getAlignmentFromAttrs(FuncAttrs);
    FuncAttrs &= ~Attribute::Alignment;
  }
  
  // Okay, if we got here, the function is syntactically valid.  Convert types
  // and do semantic checks.
  std::vector<const Type*> ParamTypeList;
  SmallVector<AttributeWithIndex, 8> Attrs;
  // FIXME : In 3.0, stop accepting zext, sext and inreg as optional function 
  // attributes.
  unsigned ObsoleteFuncAttrs = Attribute::ZExt|Attribute::SExt|Attribute::InReg;
  if (FuncAttrs & ObsoleteFuncAttrs) {
    RetAttrs |= FuncAttrs & ObsoleteFuncAttrs;
    FuncAttrs &= ~ObsoleteFuncAttrs;
  }
  
  if (RetAttrs != Attribute::None)
    Attrs.push_back(AttributeWithIndex::get(0, RetAttrs));
  
  for (unsigned i = 0, e = ArgList.size(); i != e; ++i) {
    ParamTypeList.push_back(ArgList[i].Type);
    if (ArgList[i].Attrs != Attribute::None)
      Attrs.push_back(AttributeWithIndex::get(i+1, ArgList[i].Attrs));
  }

  if (FuncAttrs != Attribute::None)
    Attrs.push_back(AttributeWithIndex::get(~0, FuncAttrs));

  AttrListPtr PAL = AttrListPtr::get(Attrs.begin(), Attrs.end());
  
  const FunctionType *FT = FunctionType::get(RetType, ParamTypeList, isVarArg);
  const PointerType *PFT = PointerType::getUnqual(FT);

  Fn = 0;
  if (!FunctionName.empty()) {
    // If this was a definition of a forward reference, remove the definition
    // from the forward reference table and fill in the forward ref.
    std::map<std::string, std::pair<GlobalValue*, LocTy> >::iterator FRVI =
      ForwardRefVals.find(FunctionName);
    if (FRVI != ForwardRefVals.end()) {
      Fn = M->getFunction(FunctionName);
      ForwardRefVals.erase(FRVI);
    } else if ((Fn = M->getFunction(FunctionName))) {
      // If this function already exists in the symbol table, then it is
      // multiply defined.  We accept a few cases for old backwards compat.
      // FIXME: Remove this stuff for LLVM 3.0.
      if (Fn->getType() != PFT || Fn->getAttributes() != PAL ||
          (!Fn->isDeclaration() && isDefine)) {
        // If the redefinition has different type or different attributes,
        // reject it.  If both have bodies, reject it.
        return Error(NameLoc, "invalid redefinition of function '" +
                     FunctionName + "'");
      } else if (Fn->isDeclaration()) {
        // Make sure to strip off any argument names so we can't get conflicts.
        for (Function::arg_iterator AI = Fn->arg_begin(), AE = Fn->arg_end();
             AI != AE; ++AI)
          AI->setName("");
      }
    }
    
  } else if (FunctionName.empty()) {
    // If this is a definition of a forward referenced function, make sure the
    // types agree.
    std::map<unsigned, std::pair<GlobalValue*, LocTy> >::iterator I
      = ForwardRefValIDs.find(NumberedVals.size());
    if (I != ForwardRefValIDs.end()) {
      Fn = cast<Function>(I->second.first);
      if (Fn->getType() != PFT)
        return Error(NameLoc, "type of definition and forward reference of '@" +
                     utostr(NumberedVals.size()) +"' disagree");
      ForwardRefValIDs.erase(I);
    }
  }

  if (Fn == 0)
    Fn = Function::Create(FT, GlobalValue::ExternalLinkage, FunctionName, M);
  else // Move the forward-reference to the correct spot in the module.
    M->getFunctionList().splice(M->end(), M->getFunctionList(), Fn);

  if (FunctionName.empty())
    NumberedVals.push_back(Fn);
  
  Fn->setLinkage((GlobalValue::LinkageTypes)Linkage);
  Fn->setVisibility((GlobalValue::VisibilityTypes)Visibility);
  Fn->setCallingConv(CC);
  Fn->setAttributes(PAL);
  Fn->setAlignment(Alignment);
  Fn->setSection(Section);
  if (!GC.empty()) Fn->setGC(GC.c_str());
    
  // Add all of the arguments we parsed to the function.
  Function::arg_iterator ArgIt = Fn->arg_begin();
  for (unsigned i = 0, e = ArgList.size(); i != e; ++i, ++ArgIt) {
    // If the argument has a name, insert it into the argument symbol table.
    if (ArgList[i].Name.empty()) continue;
    
    // Set the name, if it conflicted, it will be auto-renamed.
    ArgIt->setName(ArgList[i].Name);
    
    if (ArgIt->getNameStr() != ArgList[i].Name)
      return Error(ArgList[i].Loc, "redefinition of argument '%" +
                   ArgList[i].Name + "'");
  }
  
  return false;
}


/// ParseFunctionBody
///   ::= '{' BasicBlock+ '}'
///   ::= 'begin' BasicBlock+ 'end'  // FIXME: remove in LLVM 3.0
///
bool LLParser::ParseFunctionBody(Function &Fn) {
  if (Lex.getKind() != lltok::lbrace && Lex.getKind() != lltok::kw_begin)
    return TokError("expected '{' in function body");
  Lex.Lex();  // eat the {.
  
  PerFunctionState PFS(*this, Fn);
  
  while (Lex.getKind() != lltok::rbrace && Lex.getKind() != lltok::kw_end)
    if (ParseBasicBlock(PFS)) return true;
  
  // Eat the }.
  Lex.Lex();
  
  // Verify function is ok.
  return PFS.VerifyFunctionComplete();
}

/// ParseBasicBlock
///   ::= LabelStr? Instruction*
bool LLParser::ParseBasicBlock(PerFunctionState &PFS) {
  // If this basic block starts out with a name, remember it.
  std::string Name;
  LocTy NameLoc = Lex.getLoc();
  if (Lex.getKind() == lltok::LabelStr) {
    Name = Lex.getStrVal();
    Lex.Lex();
  }
  
  BasicBlock *BB = PFS.DefineBB(Name, NameLoc);
  if (BB == 0) return true;
  
  std::string NameStr;
  
  // Parse the instructions in this block until we get a terminator.
  Instruction *Inst;
  do {
    // This instruction may have three possibilities for a name: a) none
    // specified, b) name specified "%foo =", c) number specified: "%4 =".
    LocTy NameLoc = Lex.getLoc();
    int NameID = -1;
    NameStr = "";
    
    if (Lex.getKind() == lltok::LocalVarID) {
      NameID = Lex.getUIntVal();
      Lex.Lex();
      if (ParseToken(lltok::equal, "expected '=' after instruction id"))
        return true;
    } else if (Lex.getKind() == lltok::LocalVar ||
               // FIXME: REMOVE IN LLVM 3.0
               Lex.getKind() == lltok::StringConstant) {
      NameStr = Lex.getStrVal();
      Lex.Lex();
      if (ParseToken(lltok::equal, "expected '=' after instruction name"))
        return true;
    }
    
    if (ParseInstruction(Inst, BB, PFS)) return true;
    
    BB->getInstList().push_back(Inst);

    // Set the name on the instruction.
    if (PFS.SetInstName(NameID, NameStr, NameLoc, Inst)) return true;
  } while (!isa<TerminatorInst>(Inst));
  
  return false;
}

//===----------------------------------------------------------------------===//
// Instruction Parsing.
//===----------------------------------------------------------------------===//

/// ParseInstruction - Parse one of the many different instructions.
///
bool LLParser::ParseInstruction(Instruction *&Inst, BasicBlock *BB,
                                PerFunctionState &PFS) {
  lltok::Kind Token = Lex.getKind();
  if (Token == lltok::Eof)
    return TokError("found end of file when expecting more instructions");
  LocTy Loc = Lex.getLoc();
  Lex.Lex();  // Eat the keyword.
  
  switch (Token) {
  default:                    return Error(Loc, "expected instruction opcode");
  // Terminator Instructions.
  case lltok::kw_unwind:      Inst = new UnwindInst(); return false;
  case lltok::kw_unreachable: Inst = new UnreachableInst(); return false;
  case lltok::kw_ret:         return ParseRet(Inst, BB, PFS);
  case lltok::kw_br:          return ParseBr(Inst, PFS);
  case lltok::kw_switch:      return ParseSwitch(Inst, PFS);
  case lltok::kw_invoke:      return ParseInvoke(Inst, PFS);
  // Binary Operators.
  case lltok::kw_add:
  case lltok::kw_sub:
  case lltok::kw_mul:    return ParseArithmetic(Inst, PFS, Lex.getUIntVal(), 0);
      
  case lltok::kw_udiv:
  case lltok::kw_sdiv:
  case lltok::kw_urem:
  case lltok::kw_srem:   return ParseArithmetic(Inst, PFS, Lex.getUIntVal(), 1);
  case lltok::kw_fdiv:
  case lltok::kw_frem:   return ParseArithmetic(Inst, PFS, Lex.getUIntVal(), 2);
  case lltok::kw_shl:
  case lltok::kw_lshr:
  case lltok::kw_ashr:
  case lltok::kw_and:
  case lltok::kw_or:
  case lltok::kw_xor:    return ParseLogical(Inst, PFS, Lex.getUIntVal());
  case lltok::kw_icmp:
  case lltok::kw_fcmp:
  case lltok::kw_vicmp:
  case lltok::kw_vfcmp:  return ParseCompare(Inst, PFS, Lex.getUIntVal());
  // Casts.
  case lltok::kw_trunc:
  case lltok::kw_zext:
  case lltok::kw_sext:
  case lltok::kw_fptrunc:
  case lltok::kw_fpext:
  case lltok::kw_bitcast:
  case lltok::kw_uitofp:
  case lltok::kw_sitofp:
  case lltok::kw_fptoui:
  case lltok::kw_fptosi: 
  case lltok::kw_inttoptr:
  case lltok::kw_ptrtoint:       return ParseCast(Inst, PFS, Lex.getUIntVal());
  // Other.
  case lltok::kw_select:         return ParseSelect(Inst, PFS);
  case lltok::kw_va_arg:         return ParseVA_Arg(Inst, PFS);
  case lltok::kw_extractelement: return ParseExtractElement(Inst, PFS);
  case lltok::kw_insertelement:  return ParseInsertElement(Inst, PFS);
  case lltok::kw_shufflevector:  return ParseShuffleVector(Inst, PFS);
  case lltok::kw_phi:            return ParsePHI(Inst, PFS);
  case lltok::kw_call:           return ParseCall(Inst, PFS, false);
  case lltok::kw_tail:           return ParseCall(Inst, PFS, true);
  // Memory.
  case lltok::kw_alloca:
  case lltok::kw_malloc:         return ParseAlloc(Inst, PFS, Lex.getUIntVal());
  case lltok::kw_free:           return ParseFree(Inst, PFS);
  case lltok::kw_load:           return ParseLoad(Inst, PFS, false);
  case lltok::kw_store:          return ParseStore(Inst, PFS, false);
  case lltok::kw_volatile:
    if (EatIfPresent(lltok::kw_load))
      return ParseLoad(Inst, PFS, true);
    else if (EatIfPresent(lltok::kw_store))
      return ParseStore(Inst, PFS, true);
    else
      return TokError("expected 'load' or 'store'");
  case lltok::kw_getresult:     return ParseGetResult(Inst, PFS);
  case lltok::kw_getelementptr: return ParseGetElementPtr(Inst, PFS);
  case lltok::kw_extractvalue:  return ParseExtractValue(Inst, PFS);
  case lltok::kw_insertvalue:   return ParseInsertValue(Inst, PFS);
  }
}

/// ParseCmpPredicate - Parse an integer or fp predicate, based on Kind.
bool LLParser::ParseCmpPredicate(unsigned &P, unsigned Opc) {
  // FIXME: REMOVE vicmp/vfcmp!
  if (Opc == Instruction::FCmp || Opc == Instruction::VFCmp) {
    switch (Lex.getKind()) {
    default: TokError("expected fcmp predicate (e.g. 'oeq')");
    case lltok::kw_oeq: P = CmpInst::FCMP_OEQ; break;
    case lltok::kw_one: P = CmpInst::FCMP_ONE; break;
    case lltok::kw_olt: P = CmpInst::FCMP_OLT; break;
    case lltok::kw_ogt: P = CmpInst::FCMP_OGT; break;
    case lltok::kw_ole: P = CmpInst::FCMP_OLE; break;
    case lltok::kw_oge: P = CmpInst::FCMP_OGE; break;
    case lltok::kw_ord: P = CmpInst::FCMP_ORD; break;
    case lltok::kw_uno: P = CmpInst::FCMP_UNO; break;
    case lltok::kw_ueq: P = CmpInst::FCMP_UEQ; break;
    case lltok::kw_une: P = CmpInst::FCMP_UNE; break;
    case lltok::kw_ult: P = CmpInst::FCMP_ULT; break;
    case lltok::kw_ugt: P = CmpInst::FCMP_UGT; break;
    case lltok::kw_ule: P = CmpInst::FCMP_ULE; break;
    case lltok::kw_uge: P = CmpInst::FCMP_UGE; break;
    case lltok::kw_true: P = CmpInst::FCMP_TRUE; break;
    case lltok::kw_false: P = CmpInst::FCMP_FALSE; break;
    }
  } else {
    switch (Lex.getKind()) {
    default: TokError("expected icmp predicate (e.g. 'eq')");
    case lltok::kw_eq:  P = CmpInst::ICMP_EQ; break;
    case lltok::kw_ne:  P = CmpInst::ICMP_NE; break;
    case lltok::kw_slt: P = CmpInst::ICMP_SLT; break;
    case lltok::kw_sgt: P = CmpInst::ICMP_SGT; break;
    case lltok::kw_sle: P = CmpInst::ICMP_SLE; break;
    case lltok::kw_sge: P = CmpInst::ICMP_SGE; break;
    case lltok::kw_ult: P = CmpInst::ICMP_ULT; break;
    case lltok::kw_ugt: P = CmpInst::ICMP_UGT; break;
    case lltok::kw_ule: P = CmpInst::ICMP_ULE; break;
    case lltok::kw_uge: P = CmpInst::ICMP_UGE; break;
    }
  }
  Lex.Lex();
  return false;
}

//===----------------------------------------------------------------------===//
// Terminator Instructions.
//===----------------------------------------------------------------------===//

/// ParseRet - Parse a return instruction.
///   ::= 'ret' void
///   ::= 'ret' TypeAndValue
///   ::= 'ret' TypeAndValue (',' TypeAndValue)+  [[obsolete: LLVM 3.0]]
bool LLParser::ParseRet(Instruction *&Inst, BasicBlock *BB,
                        PerFunctionState &PFS) {
  PATypeHolder Ty(Type::VoidTy);
  if (ParseType(Ty)) return true;
  
  if (Ty == Type::VoidTy) {
    Inst = ReturnInst::Create();
    return false;
  }
  
  Value *RV;
  if (ParseValue(Ty, RV, PFS)) return true;
  
  // The normal case is one return value.
  if (Lex.getKind() == lltok::comma) {
    // FIXME: LLVM 3.0 remove MRV support for 'ret i32 1, i32 2', requiring use
    // of 'ret {i32,i32} {i32 1, i32 2}'
    SmallVector<Value*, 8> RVs;
    RVs.push_back(RV);
    
    while (EatIfPresent(lltok::comma)) {
      if (ParseTypeAndValue(RV, PFS)) return true;
      RVs.push_back(RV);
    }

    RV = UndefValue::get(PFS.getFunction().getReturnType());
    for (unsigned i = 0, e = RVs.size(); i != e; ++i) {
      Instruction *I = InsertValueInst::Create(RV, RVs[i], i, "mrv");
      BB->getInstList().push_back(I);
      RV = I;
    }
  }
  Inst = ReturnInst::Create(RV);
  return false;
}


/// ParseBr
///   ::= 'br' TypeAndValue
///   ::= 'br' TypeAndValue ',' TypeAndValue ',' TypeAndValue
bool LLParser::ParseBr(Instruction *&Inst, PerFunctionState &PFS) {
  LocTy Loc, Loc2;
  Value *Op0, *Op1, *Op2;
  if (ParseTypeAndValue(Op0, Loc, PFS)) return true;
  
  if (BasicBlock *BB = dyn_cast<BasicBlock>(Op0)) {
    Inst = BranchInst::Create(BB);
    return false;
  }
  
  if (Op0->getType() != Type::Int1Ty)
    return Error(Loc, "branch condition must have 'i1' type");
    
  if (ParseToken(lltok::comma, "expected ',' after branch condition") ||
      ParseTypeAndValue(Op1, Loc, PFS) ||
      ParseToken(lltok::comma, "expected ',' after true destination") ||
      ParseTypeAndValue(Op2, Loc2, PFS))
    return true;
  
  if (!isa<BasicBlock>(Op1))
    return Error(Loc, "true destination of branch must be a basic block");
  if (!isa<BasicBlock>(Op2))
    return Error(Loc2, "true destination of branch must be a basic block");
    
  Inst = BranchInst::Create(cast<BasicBlock>(Op1), cast<BasicBlock>(Op2), Op0);
  return false;
}

/// ParseSwitch
///  Instruction
///    ::= 'switch' TypeAndValue ',' TypeAndValue '[' JumpTable ']'
///  JumpTable
///    ::= (TypeAndValue ',' TypeAndValue)*
bool LLParser::ParseSwitch(Instruction *&Inst, PerFunctionState &PFS) {
  LocTy CondLoc, BBLoc;
  Value *Cond, *DefaultBB;
  if (ParseTypeAndValue(Cond, CondLoc, PFS) ||
      ParseToken(lltok::comma, "expected ',' after switch condition") ||
      ParseTypeAndValue(DefaultBB, BBLoc, PFS) ||
      ParseToken(lltok::lsquare, "expected '[' with switch table"))
    return true;

  if (!isa<IntegerType>(Cond->getType()))
    return Error(CondLoc, "switch condition must have integer type");
  if (!isa<BasicBlock>(DefaultBB))
    return Error(BBLoc, "default destination must be a basic block");
  
  // Parse the jump table pairs.
  SmallPtrSet<Value*, 32> SeenCases;
  SmallVector<std::pair<ConstantInt*, BasicBlock*>, 32> Table;
  while (Lex.getKind() != lltok::rsquare) {
    Value *Constant, *DestBB;
    
    if (ParseTypeAndValue(Constant, CondLoc, PFS) ||
        ParseToken(lltok::comma, "expected ',' after case value") ||
        ParseTypeAndValue(DestBB, BBLoc, PFS))
      return true;

    if (!SeenCases.insert(Constant))
      return Error(CondLoc, "duplicate case value in switch");
    if (!isa<ConstantInt>(Constant))
      return Error(CondLoc, "case value is not a constant integer");
    if (!isa<BasicBlock>(DestBB))
      return Error(BBLoc, "case destination is not a basic block");
    
    Table.push_back(std::make_pair(cast<ConstantInt>(Constant),
                                   cast<BasicBlock>(DestBB)));
  }
  
  Lex.Lex();  // Eat the ']'.
  
  SwitchInst *SI = SwitchInst::Create(Cond, cast<BasicBlock>(DefaultBB),
                                      Table.size());
  for (unsigned i = 0, e = Table.size(); i != e; ++i)
    SI->addCase(Table[i].first, Table[i].second);
  Inst = SI;
  return false;
}

/// ParseInvoke
///   ::= 'invoke' OptionalCallingConv OptionalAttrs Type Value ParamList
///       OptionalAttrs 'to' TypeAndValue 'unwind' TypeAndValue
bool LLParser::ParseInvoke(Instruction *&Inst, PerFunctionState &PFS) {
  LocTy CallLoc = Lex.getLoc();
  unsigned CC, RetAttrs, FnAttrs;
  PATypeHolder RetType(Type::VoidTy);
  LocTy RetTypeLoc;
  ValID CalleeID;
  SmallVector<ParamInfo, 16> ArgList;

  Value *NormalBB, *UnwindBB;
  if (ParseOptionalCallingConv(CC) ||
      ParseOptionalAttrs(RetAttrs, 1) ||
      ParseType(RetType, RetTypeLoc) ||
      ParseValID(CalleeID) ||
      ParseParameterList(ArgList, PFS) ||
      ParseOptionalAttrs(FnAttrs, 2) ||
      ParseToken(lltok::kw_to, "expected 'to' in invoke") ||
      ParseTypeAndValue(NormalBB, PFS) ||
      ParseToken(lltok::kw_unwind, "expected 'unwind' in invoke") ||
      ParseTypeAndValue(UnwindBB, PFS))
    return true;
  
  if (!isa<BasicBlock>(NormalBB))
    return Error(CallLoc, "normal destination is not a basic block");
  if (!isa<BasicBlock>(UnwindBB))
    return Error(CallLoc, "unwind destination is not a basic block");
  
  // If RetType is a non-function pointer type, then this is the short syntax
  // for the call, which means that RetType is just the return type.  Infer the
  // rest of the function argument types from the arguments that are present.
  const PointerType *PFTy = 0;
  const FunctionType *Ty = 0;
  if (!(PFTy = dyn_cast<PointerType>(RetType)) ||
      !(Ty = dyn_cast<FunctionType>(PFTy->getElementType()))) {
    // Pull out the types of all of the arguments...
    std::vector<const Type*> ParamTypes;
    for (unsigned i = 0, e = ArgList.size(); i != e; ++i)
      ParamTypes.push_back(ArgList[i].V->getType());
    
    if (!FunctionType::isValidReturnType(RetType))
      return Error(RetTypeLoc, "Invalid result type for LLVM function");
    
    Ty = FunctionType::get(RetType, ParamTypes, false);
    PFTy = PointerType::getUnqual(Ty);
  }
  
  // Look up the callee.
  Value *Callee;
  if (ConvertValIDToValue(PFTy, CalleeID, Callee, PFS)) return true;
  
  // FIXME: In LLVM 3.0, stop accepting zext, sext and inreg as optional
  // function attributes.
  unsigned ObsoleteFuncAttrs = Attribute::ZExt|Attribute::SExt|Attribute::InReg;
  if (FnAttrs & ObsoleteFuncAttrs) {
    RetAttrs |= FnAttrs & ObsoleteFuncAttrs;
    FnAttrs &= ~ObsoleteFuncAttrs;
  }
  
  // Set up the Attributes for the function.
  SmallVector<AttributeWithIndex, 8> Attrs;
  if (RetAttrs != Attribute::None)
    Attrs.push_back(AttributeWithIndex::get(0, RetAttrs));
  
  SmallVector<Value*, 8> Args;
  
  // Loop through FunctionType's arguments and ensure they are specified
  // correctly.  Also, gather any parameter attributes.
  FunctionType::param_iterator I = Ty->param_begin();
  FunctionType::param_iterator E = Ty->param_end();
  for (unsigned i = 0, e = ArgList.size(); i != e; ++i) {
    const Type *ExpectedTy = 0;
    if (I != E) {
      ExpectedTy = *I++;
    } else if (!Ty->isVarArg()) {
      return Error(ArgList[i].Loc, "too many arguments specified");
    }
    
    if (ExpectedTy && ExpectedTy != ArgList[i].V->getType())
      return Error(ArgList[i].Loc, "argument is not of expected type '" +
                   ExpectedTy->getDescription() + "'");
    Args.push_back(ArgList[i].V);
    if (ArgList[i].Attrs != Attribute::None)
      Attrs.push_back(AttributeWithIndex::get(i+1, ArgList[i].Attrs));
  }
  
  if (I != E)
    return Error(CallLoc, "not enough parameters specified for call");
  
  if (FnAttrs != Attribute::None)
    Attrs.push_back(AttributeWithIndex::get(~0, FnAttrs));
  
  // Finish off the Attributes and check them
  AttrListPtr PAL = AttrListPtr::get(Attrs.begin(), Attrs.end());
  
  InvokeInst *II = InvokeInst::Create(Callee, cast<BasicBlock>(NormalBB),
                                      cast<BasicBlock>(UnwindBB),
                                      Args.begin(), Args.end());
  II->setCallingConv(CC);
  II->setAttributes(PAL);
  Inst = II;
  return false;
}



//===----------------------------------------------------------------------===//
// Binary Operators.
//===----------------------------------------------------------------------===//

/// ParseArithmetic
///  ::= ArithmeticOps TypeAndValue ',' Value
///
/// If OperandType is 0, then any FP or integer operand is allowed.  If it is 1,
/// then any integer operand is allowed, if it is 2, any fp operand is allowed.
bool LLParser::ParseArithmetic(Instruction *&Inst, PerFunctionState &PFS,
                               unsigned Opc, unsigned OperandType) {
  LocTy Loc; Value *LHS, *RHS;
  if (ParseTypeAndValue(LHS, Loc, PFS) ||
      ParseToken(lltok::comma, "expected ',' in arithmetic operation") ||
      ParseValue(LHS->getType(), RHS, PFS))
    return true;

  bool Valid;
  switch (OperandType) {
  default: assert(0 && "Unknown operand type!");
  case 0: // int or FP.
    Valid = LHS->getType()->isIntOrIntVector() ||
            LHS->getType()->isFPOrFPVector();
    break;
  case 1: Valid = LHS->getType()->isIntOrIntVector(); break;
  case 2: Valid = LHS->getType()->isFPOrFPVector(); break;
  }
  
  if (!Valid)
    return Error(Loc, "invalid operand type for instruction");
  
  Inst = BinaryOperator::Create((Instruction::BinaryOps)Opc, LHS, RHS);
  return false;
}

/// ParseLogical
///  ::= ArithmeticOps TypeAndValue ',' Value {
bool LLParser::ParseLogical(Instruction *&Inst, PerFunctionState &PFS,
                            unsigned Opc) {
  LocTy Loc; Value *LHS, *RHS;
  if (ParseTypeAndValue(LHS, Loc, PFS) ||
      ParseToken(lltok::comma, "expected ',' in logical operation") ||
      ParseValue(LHS->getType(), RHS, PFS))
    return true;

  if (!LHS->getType()->isIntOrIntVector())
    return Error(Loc,"instruction requires integer or integer vector operands");

  Inst = BinaryOperator::Create((Instruction::BinaryOps)Opc, LHS, RHS);
  return false;
}


/// ParseCompare
///  ::= 'icmp' IPredicates TypeAndValue ',' Value
///  ::= 'fcmp' FPredicates TypeAndValue ',' Value
///  ::= 'vicmp' IPredicates TypeAndValue ',' Value
///  ::= 'vfcmp' FPredicates TypeAndValue ',' Value
bool LLParser::ParseCompare(Instruction *&Inst, PerFunctionState &PFS,
                            unsigned Opc) {
  // Parse the integer/fp comparison predicate.
  LocTy Loc;
  unsigned Pred;
  Value *LHS, *RHS;
  if (ParseCmpPredicate(Pred, Opc) ||
      ParseTypeAndValue(LHS, Loc, PFS) ||
      ParseToken(lltok::comma, "expected ',' after compare value") ||
      ParseValue(LHS->getType(), RHS, PFS))
    return true;
  
  if (Opc == Instruction::FCmp) {
    if (!LHS->getType()->isFPOrFPVector())
      return Error(Loc, "fcmp requires floating point operands");
    Inst = new FCmpInst(CmpInst::Predicate(Pred), LHS, RHS);
  } else if (Opc == Instruction::ICmp) {
    if (!LHS->getType()->isIntOrIntVector() &&
        !isa<PointerType>(LHS->getType()))
      return Error(Loc, "icmp requires integer operands");
    Inst = new ICmpInst(CmpInst::Predicate(Pred), LHS, RHS);
  } else if (Opc == Instruction::VFCmp) {
    if (!LHS->getType()->isFPOrFPVector() || !isa<VectorType>(LHS->getType()))
      return Error(Loc, "vfcmp requires vector floating point operands");
    Inst = new VFCmpInst(CmpInst::Predicate(Pred), LHS, RHS);
  } else if (Opc == Instruction::VICmp) {
    if (!LHS->getType()->isIntOrIntVector() || !isa<VectorType>(LHS->getType()))
      return Error(Loc, "vicmp requires vector floating point operands");
    Inst = new VICmpInst(CmpInst::Predicate(Pred), LHS, RHS);
  }
  return false;
}

//===----------------------------------------------------------------------===//
// Other Instructions.
//===----------------------------------------------------------------------===//


/// ParseCast
///   ::= CastOpc TypeAndValue 'to' Type
bool LLParser::ParseCast(Instruction *&Inst, PerFunctionState &PFS,
                         unsigned Opc) {
  LocTy Loc;  Value *Op;
  PATypeHolder DestTy(Type::VoidTy);
  if (ParseTypeAndValue(Op, Loc, PFS) ||
      ParseToken(lltok::kw_to, "expected 'to' after cast value") ||
      ParseType(DestTy))
    return true;
  
  if (!CastInst::castIsValid((Instruction::CastOps)Opc, Op, DestTy))
    return Error(Loc, "invalid cast opcode for cast from '" +
                 Op->getType()->getDescription() + "' to '" +
                 DestTy->getDescription() + "'");
  Inst = CastInst::Create((Instruction::CastOps)Opc, Op, DestTy);
  return false;
}

/// ParseSelect
///   ::= 'select' TypeAndValue ',' TypeAndValue ',' TypeAndValue
bool LLParser::ParseSelect(Instruction *&Inst, PerFunctionState &PFS) {
  LocTy Loc;
  Value *Op0, *Op1, *Op2;
  if (ParseTypeAndValue(Op0, Loc, PFS) ||
      ParseToken(lltok::comma, "expected ',' after select condition") ||
      ParseTypeAndValue(Op1, PFS) ||
      ParseToken(lltok::comma, "expected ',' after select value") ||
      ParseTypeAndValue(Op2, PFS))
    return true;
  
  if (const char *Reason = SelectInst::areInvalidOperands(Op0, Op1, Op2))
    return Error(Loc, Reason);
  
  Inst = SelectInst::Create(Op0, Op1, Op2);
  return false;
}

/// ParseVA_Arg
///   ::= 'va_arg' TypeAndValue ',' Type
bool LLParser::ParseVA_Arg(Instruction *&Inst, PerFunctionState &PFS) {
  Value *Op;
  PATypeHolder EltTy(Type::VoidTy);
  LocTy TypeLoc;
  if (ParseTypeAndValue(Op, PFS) ||
      ParseToken(lltok::comma, "expected ',' after vaarg operand") ||
      ParseType(EltTy, TypeLoc))
    return true;
  
  if (!EltTy->isFirstClassType())
    return Error(TypeLoc, "va_arg requires operand with first class type");

  Inst = new VAArgInst(Op, EltTy);
  return false;
}

/// ParseExtractElement
///   ::= 'extractelement' TypeAndValue ',' TypeAndValue
bool LLParser::ParseExtractElement(Instruction *&Inst, PerFunctionState &PFS) {
  LocTy Loc;
  Value *Op0, *Op1;
  if (ParseTypeAndValue(Op0, Loc, PFS) ||
      ParseToken(lltok::comma, "expected ',' after extract value") ||
      ParseTypeAndValue(Op1, PFS))
    return true;
  
  if (!ExtractElementInst::isValidOperands(Op0, Op1))
    return Error(Loc, "invalid extractelement operands");
  
  Inst = new ExtractElementInst(Op0, Op1);
  return false;
}

/// ParseInsertElement
///   ::= 'insertelement' TypeAndValue ',' TypeAndValue ',' TypeAndValue
bool LLParser::ParseInsertElement(Instruction *&Inst, PerFunctionState &PFS) {
  LocTy Loc;
  Value *Op0, *Op1, *Op2;
  if (ParseTypeAndValue(Op0, Loc, PFS) ||
      ParseToken(lltok::comma, "expected ',' after insertelement value") ||
      ParseTypeAndValue(Op1, PFS) ||
      ParseToken(lltok::comma, "expected ',' after insertelement value") ||
      ParseTypeAndValue(Op2, PFS))
    return true;
  
  if (!InsertElementInst::isValidOperands(Op0, Op1, Op2))
    return Error(Loc, "invalid extractelement operands");
  
  Inst = InsertElementInst::Create(Op0, Op1, Op2);
  return false;
}

/// ParseShuffleVector
///   ::= 'shufflevector' TypeAndValue ',' TypeAndValue ',' TypeAndValue
bool LLParser::ParseShuffleVector(Instruction *&Inst, PerFunctionState &PFS) {
  LocTy Loc;
  Value *Op0, *Op1, *Op2;
  if (ParseTypeAndValue(Op0, Loc, PFS) ||
      ParseToken(lltok::comma, "expected ',' after shuffle mask") ||
      ParseTypeAndValue(Op1, PFS) ||
      ParseToken(lltok::comma, "expected ',' after shuffle value") ||
      ParseTypeAndValue(Op2, PFS))
    return true;
  
  if (!ShuffleVectorInst::isValidOperands(Op0, Op1, Op2))
    return Error(Loc, "invalid extractelement operands");
  
  Inst = new ShuffleVectorInst(Op0, Op1, Op2);
  return false;
}

/// ParsePHI
///   ::= 'phi' Type '[' Value ',' Value ']' (',' '[' Value ',' Valueß ']')*
bool LLParser::ParsePHI(Instruction *&Inst, PerFunctionState &PFS) {
  PATypeHolder Ty(Type::VoidTy);
  Value *Op0, *Op1;
  LocTy TypeLoc = Lex.getLoc();
  
  if (ParseType(Ty) ||
      ParseToken(lltok::lsquare, "expected '[' in phi value list") ||
      ParseValue(Ty, Op0, PFS) ||
      ParseToken(lltok::comma, "expected ',' after insertelement value") ||
      ParseValue(Type::LabelTy, Op1, PFS) ||
      ParseToken(lltok::rsquare, "expected ']' in phi value list"))
    return true;
 
  SmallVector<std::pair<Value*, BasicBlock*>, 16> PHIVals;
  while (1) {
    PHIVals.push_back(std::make_pair(Op0, cast<BasicBlock>(Op1)));
    
    if (!EatIfPresent(lltok::comma))
      break;

    if (ParseToken(lltok::lsquare, "expected '[' in phi value list") ||
        ParseValue(Ty, Op0, PFS) ||
        ParseToken(lltok::comma, "expected ',' after insertelement value") ||
        ParseValue(Type::LabelTy, Op1, PFS) ||
        ParseToken(lltok::rsquare, "expected ']' in phi value list"))
      return true;
  }
  
  if (!Ty->isFirstClassType())
    return Error(TypeLoc, "phi node must have first class type");

  PHINode *PN = PHINode::Create(Ty);
  PN->reserveOperandSpace(PHIVals.size());
  for (unsigned i = 0, e = PHIVals.size(); i != e; ++i)
    PN->addIncoming(PHIVals[i].first, PHIVals[i].second);
  Inst = PN;
  return false;
}

/// ParseCall
///   ::= 'tail'? 'call' OptionalCallingConv OptionalAttrs Type Value
///       ParameterList OptionalAttrs
bool LLParser::ParseCall(Instruction *&Inst, PerFunctionState &PFS,
                         bool isTail) {
  unsigned CC, RetAttrs, FnAttrs;
  PATypeHolder RetType(Type::VoidTy);
  LocTy RetTypeLoc;
  ValID CalleeID;
  SmallVector<ParamInfo, 16> ArgList;
  LocTy CallLoc = Lex.getLoc();
  
  if ((isTail && ParseToken(lltok::kw_call, "expected 'tail call'")) ||
      ParseOptionalCallingConv(CC) ||
      ParseOptionalAttrs(RetAttrs, 1) ||
      ParseType(RetType, RetTypeLoc) ||
      ParseValID(CalleeID) ||
      ParseParameterList(ArgList, PFS) ||
      ParseOptionalAttrs(FnAttrs, 2))
    return true;
  
  // If RetType is a non-function pointer type, then this is the short syntax
  // for the call, which means that RetType is just the return type.  Infer the
  // rest of the function argument types from the arguments that are present.
  const PointerType *PFTy = 0;
  const FunctionType *Ty = 0;
  if (!(PFTy = dyn_cast<PointerType>(RetType)) ||
      !(Ty = dyn_cast<FunctionType>(PFTy->getElementType()))) {
    // Pull out the types of all of the arguments...
    std::vector<const Type*> ParamTypes;
    for (unsigned i = 0, e = ArgList.size(); i != e; ++i)
      ParamTypes.push_back(ArgList[i].V->getType());
    
    if (!FunctionType::isValidReturnType(RetType))
      return Error(RetTypeLoc, "Invalid result type for LLVM function");
    
    Ty = FunctionType::get(RetType, ParamTypes, false);
    PFTy = PointerType::getUnqual(Ty);
  }
  
  // Look up the callee.
  Value *Callee;
  if (ConvertValIDToValue(PFTy, CalleeID, Callee, PFS)) return true;
  
  // Check for call to invalid intrinsic to avoid crashing later.
  if (Function *F = dyn_cast<Function>(Callee)) {
    if (F->hasName() && F->getNameLen() >= 5 &&
        !strncmp(F->getValueName()->getKeyData(), "llvm.", 5) &&
        !F->getIntrinsicID(true))
      return Error(CallLoc, "Call to invalid LLVM intrinsic function '" +
                   F->getNameStr() + "'");
  }
  
  // FIXME: In LLVM 3.0, stop accepting zext, sext and inreg as optional
  // function attributes.
  unsigned ObsoleteFuncAttrs = Attribute::ZExt|Attribute::SExt|Attribute::InReg;
  if (FnAttrs & ObsoleteFuncAttrs) {
    RetAttrs |= FnAttrs & ObsoleteFuncAttrs;
    FnAttrs &= ~ObsoleteFuncAttrs;
  }

  // Set up the Attributes for the function.
  SmallVector<AttributeWithIndex, 8> Attrs;
  if (RetAttrs != Attribute::None)
    Attrs.push_back(AttributeWithIndex::get(0, RetAttrs));
  
  SmallVector<Value*, 8> Args;
  
  // Loop through FunctionType's arguments and ensure they are specified
  // correctly.  Also, gather any parameter attributes.
  FunctionType::param_iterator I = Ty->param_begin();
  FunctionType::param_iterator E = Ty->param_end();
  for (unsigned i = 0, e = ArgList.size(); i != e; ++i) {
    const Type *ExpectedTy = 0;
    if (I != E) {
      ExpectedTy = *I++;
    } else if (!Ty->isVarArg()) {
      return Error(ArgList[i].Loc, "too many arguments specified");
    }
    
    if (ExpectedTy && ExpectedTy != ArgList[i].V->getType())
      return Error(ArgList[i].Loc, "argument is not of expected type '" +
                   ExpectedTy->getDescription() + "'");
    Args.push_back(ArgList[i].V);
    if (ArgList[i].Attrs != Attribute::None)
      Attrs.push_back(AttributeWithIndex::get(i+1, ArgList[i].Attrs));
  }
  
  if (I != E)
    return Error(CallLoc, "not enough parameters specified for call");

  if (FnAttrs != Attribute::None)
    Attrs.push_back(AttributeWithIndex::get(~0, FnAttrs));

  // Finish off the Attributes and check them
  AttrListPtr PAL = AttrListPtr::get(Attrs.begin(), Attrs.end());
  
  CallInst *CI = CallInst::Create(Callee, Args.begin(), Args.end());
  CI->setTailCall(isTail);
  CI->setCallingConv(CC);
  CI->setAttributes(PAL);
  Inst = CI;
  return false;
}

//===----------------------------------------------------------------------===//
// Memory Instructions.
//===----------------------------------------------------------------------===//

/// ParseAlloc
///   ::= 'malloc' Type (',' TypeAndValue)? (',' OptionalAlignment)?
///   ::= 'alloca' Type (',' TypeAndValue)? (',' OptionalAlignment)?
bool LLParser::ParseAlloc(Instruction *&Inst, PerFunctionState &PFS,
                          unsigned Opc) {
  PATypeHolder Ty(Type::VoidTy);
  Value *Size = 0;
  LocTy SizeLoc = 0;
  unsigned Alignment = 0;
  if (ParseType(Ty)) return true;

  if (EatIfPresent(lltok::comma)) {
    if (Lex.getKind() == lltok::kw_align) {
      if (ParseOptionalAlignment(Alignment)) return true;
    } else if (ParseTypeAndValue(Size, SizeLoc, PFS) ||
               ParseOptionalCommaAlignment(Alignment)) {
      return true;
    }
  }

  if (Size && Size->getType() != Type::Int32Ty)
    return Error(SizeLoc, "element count must be i32");

  if (Opc == Instruction::Malloc)
    Inst = new MallocInst(Ty, Size, Alignment);
  else
    Inst = new AllocaInst(Ty, Size, Alignment);
  return false;
}

/// ParseFree
///   ::= 'free' TypeAndValue
bool LLParser::ParseFree(Instruction *&Inst, PerFunctionState &PFS) {
  Value *Val; LocTy Loc;
  if (ParseTypeAndValue(Val, Loc, PFS)) return true;
  if (!isa<PointerType>(Val->getType()))
    return Error(Loc, "operand to free must be a pointer");
  Inst = new FreeInst(Val);
  return false;
}

/// ParseLoad
///   ::= 'volatile'? 'load' TypeAndValue (',' 'align' uint)?
bool LLParser::ParseLoad(Instruction *&Inst, PerFunctionState &PFS,
                         bool isVolatile) {
  Value *Val; LocTy Loc;
  unsigned Alignment;
  if (ParseTypeAndValue(Val, Loc, PFS) ||
      ParseOptionalCommaAlignment(Alignment))
    return true;

  if (!isa<PointerType>(Val->getType()) ||
      !cast<PointerType>(Val->getType())->getElementType()->isFirstClassType())
    return Error(Loc, "load operand must be a pointer to a first class type");
  
  Inst = new LoadInst(Val, "", isVolatile, Alignment);
  return false;
}

/// ParseStore
///   ::= 'volatile'? 'store' TypeAndValue ',' TypeAndValue (',' 'align' uint)?
bool LLParser::ParseStore(Instruction *&Inst, PerFunctionState &PFS,
                          bool isVolatile) {
  Value *Val, *Ptr; LocTy Loc, PtrLoc;
  unsigned Alignment;
  if (ParseTypeAndValue(Val, Loc, PFS) ||
      ParseToken(lltok::comma, "expected ',' after store operand") ||
      ParseTypeAndValue(Ptr, PtrLoc, PFS) ||
      ParseOptionalCommaAlignment(Alignment))
    return true;
  
  if (!isa<PointerType>(Ptr->getType()))
    return Error(PtrLoc, "store operand must be a pointer");
  if (!Val->getType()->isFirstClassType())
    return Error(Loc, "store operand must be a first class value");
  if (cast<PointerType>(Ptr->getType())->getElementType() != Val->getType())
    return Error(Loc, "stored value and pointer type do not match");
  
  Inst = new StoreInst(Val, Ptr, isVolatile, Alignment);
  return false;
}

/// ParseGetResult
///   ::= 'getresult' TypeAndValue ',' uint
/// FIXME: Remove support for getresult in LLVM 3.0
bool LLParser::ParseGetResult(Instruction *&Inst, PerFunctionState &PFS) {
  Value *Val; LocTy ValLoc, EltLoc;
  unsigned Element;
  if (ParseTypeAndValue(Val, ValLoc, PFS) ||
      ParseToken(lltok::comma, "expected ',' after getresult operand") ||
      ParseUInt32(Element, EltLoc))
    return true;
  
  if (!isa<StructType>(Val->getType()) && !isa<ArrayType>(Val->getType()))
    return Error(ValLoc, "getresult inst requires an aggregate operand");
  if (!ExtractValueInst::getIndexedType(Val->getType(), Element))
    return Error(EltLoc, "invalid getresult index for value");
  Inst = ExtractValueInst::Create(Val, Element);
  return false;
}

/// ParseGetElementPtr
///   ::= 'getelementptr' TypeAndValue (',' TypeAndValue)*
bool LLParser::ParseGetElementPtr(Instruction *&Inst, PerFunctionState &PFS) {
  Value *Ptr, *Val; LocTy Loc, EltLoc;
  if (ParseTypeAndValue(Ptr, Loc, PFS)) return true;
  
  if (!isa<PointerType>(Ptr->getType()))
    return Error(Loc, "base of getelementptr must be a pointer");
  
  SmallVector<Value*, 16> Indices;
  while (EatIfPresent(lltok::comma)) {
    if (ParseTypeAndValue(Val, EltLoc, PFS)) return true;
    if (!isa<IntegerType>(Val->getType()))
      return Error(EltLoc, "getelementptr index must be an integer");
    Indices.push_back(Val);
  }
  
  if (!GetElementPtrInst::getIndexedType(Ptr->getType(),
                                         Indices.begin(), Indices.end()))
    return Error(Loc, "invalid getelementptr indices");
  Inst = GetElementPtrInst::Create(Ptr, Indices.begin(), Indices.end());
  return false;
}

/// ParseExtractValue
///   ::= 'extractvalue' TypeAndValue (',' uint32)+
bool LLParser::ParseExtractValue(Instruction *&Inst, PerFunctionState &PFS) {
  Value *Val; LocTy Loc;
  SmallVector<unsigned, 4> Indices;
  if (ParseTypeAndValue(Val, Loc, PFS) ||
      ParseIndexList(Indices))
    return true;

  if (!isa<StructType>(Val->getType()) && !isa<ArrayType>(Val->getType()))
    return Error(Loc, "extractvalue operand must be array or struct");

  if (!ExtractValueInst::getIndexedType(Val->getType(), Indices.begin(),
                                        Indices.end()))
    return Error(Loc, "invalid indices for extractvalue");
  Inst = ExtractValueInst::Create(Val, Indices.begin(), Indices.end());
  return false;
}

/// ParseInsertValue
///   ::= 'insertvalue' TypeAndValue ',' TypeAndValue (',' uint32)+
bool LLParser::ParseInsertValue(Instruction *&Inst, PerFunctionState &PFS) {
  Value *Val0, *Val1; LocTy Loc0, Loc1;
  SmallVector<unsigned, 4> Indices;
  if (ParseTypeAndValue(Val0, Loc0, PFS) ||
      ParseToken(lltok::comma, "expected comma after insertvalue operand") ||
      ParseTypeAndValue(Val1, Loc1, PFS) ||
      ParseIndexList(Indices))
    return true;
  
  if (!isa<StructType>(Val0->getType()) && !isa<ArrayType>(Val0->getType()))
    return Error(Loc0, "extractvalue operand must be array or struct");
  
  if (!ExtractValueInst::getIndexedType(Val0->getType(), Indices.begin(),
                                        Indices.end()))
    return Error(Loc0, "invalid indices for insertvalue");
  Inst = InsertValueInst::Create(Val0, Val1, Indices.begin(), Indices.end());
  return false;
}
