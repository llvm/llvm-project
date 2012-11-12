//===-- Mangler.cpp - Self-contained c/asm llvm name mangler --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Unified name mangler for assembly backends.
//
//===----------------------------------------------------------------------===//

#include "llvm/Target/Mangler.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/DataLayout.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/Twine.h"
using namespace llvm;

static bool isAcceptableChar(char C, bool AllowPeriod, bool AllowUTF8) {
  if ((C < 'a' || C > 'z') &&
      (C < 'A' || C > 'Z') &&
      (C < '0' || C > '9') &&
      C != '_' && C != '$' && C != '@' &&
      !(AllowPeriod && C == '.') &&
      !(AllowUTF8 && (C & 0x80)))
    return false;
  return true;
}

static char HexDigit(int V) {
  return V < 10 ? V+'0' : V+'A'-10;
}

static void MangleLetter(SmallVectorImpl<char> &OutName, unsigned char C) {
  OutName.push_back('_');
  OutName.push_back(HexDigit(C >> 4));
  OutName.push_back(HexDigit(C & 15));
  OutName.push_back('_');
}

/// NameNeedsEscaping - Return true if the identifier \p Str needs quotes
/// for this assembler.
static bool NameNeedsEscaping(StringRef Str, const MCAsmInfo &MAI) {
  assert(!Str.empty() && "Cannot create an empty MCSymbol");
  
  // If the first character is a number and the target does not allow this, we
  // need quotes.
  if (!MAI.doesAllowNameToStartWithDigit() && Str[0] >= '0' && Str[0] <= '9')
    return true;
  
  // If any of the characters in the string is an unacceptable character, force
  // quotes.
  bool AllowPeriod = MAI.doesAllowPeriodsInName();
  bool AllowUTF8 = MAI.doesAllowUTF8();
  for (unsigned i = 0, e = Str.size(); i != e; ++i)
    if (!isAcceptableChar(Str[i], AllowPeriod, AllowUTF8))
      return true;
  return false;
}

/// appendMangledName - Add the specified string in mangled form if it uses
/// any unusual characters.
static void appendMangledName(SmallVectorImpl<char> &OutName, StringRef Str,
                              const MCAsmInfo &MAI) {
  // The first character is not allowed to be a number unless the target
  // explicitly allows it.
  if (!MAI.doesAllowNameToStartWithDigit() && Str[0] >= '0' && Str[0] <= '9') {
    MangleLetter(OutName, Str[0]);
    Str = Str.substr(1);
  }

  bool AllowPeriod = MAI.doesAllowPeriodsInName();
  bool AllowUTF8 = MAI.doesAllowUTF8();
  for (unsigned i = 0, e = Str.size(); i != e; ++i) {
    if (!isAcceptableChar(Str[i], AllowPeriod, AllowUTF8))
      MangleLetter(OutName, Str[i]);
    else
      OutName.push_back(Str[i]);
  }
}


/// appendMangledQuotedName - On systems that support quoted symbols, we still
/// have to escape some (obscure) characters like " and \n which would break the
/// assembler's lexing.
static void appendMangledQuotedName(SmallVectorImpl<char> &OutName,
                                   StringRef Str) {
  for (unsigned i = 0, e = Str.size(); i != e; ++i) {
    if (Str[i] == '"' || Str[i] == '\n')
      MangleLetter(OutName, Str[i]);
    else
      OutName.push_back(Str[i]);
  }
}


/// getNameWithPrefix - Fill OutName with the name of the appropriate prefix
/// and the specified name as the global variable name.  GVName must not be
/// empty.
void Mangler::getNameWithPrefix(SmallVectorImpl<char> &OutName,
                                const Twine &GVName, ManglerPrefixTy PrefixTy) {
  SmallString<256> TmpData;
  StringRef Name = GVName.toStringRef(TmpData);
  assert(!Name.empty() && "getNameWithPrefix requires non-empty name");
  
  const MCAsmInfo &MAI = Context.getAsmInfo();
  
  // If the global name is not led with \1, add the appropriate prefixes.
  if (Name[0] == '\1') {
    Name = Name.substr(1);
  } else {
    if (PrefixTy == Mangler::Private) {
      const char *Prefix = MAI.getPrivateGlobalPrefix();
      OutName.append(Prefix, Prefix+strlen(Prefix));
    } else if (PrefixTy == Mangler::LinkerPrivate) {
      const char *Prefix = MAI.getLinkerPrivateGlobalPrefix();
      OutName.append(Prefix, Prefix+strlen(Prefix));
    }

    const char *Prefix = MAI.getGlobalPrefix();
    if (Prefix[0] == 0)
      ; // Common noop, no prefix.
    else if (Prefix[1] == 0)
      OutName.push_back(Prefix[0]);  // Common, one character prefix.
    else
      OutName.append(Prefix, Prefix+strlen(Prefix)); // Arbitrary length prefix.
  }
  
  // If this is a simple string that doesn't need escaping, just append it.
  if (!NameNeedsEscaping(Name, MAI) ||
      // If quotes are supported, they can be used unless the string contains
      // a quote or newline.
      (MAI.doesAllowQuotesInName() &&
       Name.find_first_of("\n\"") == StringRef::npos)) {
    OutName.append(Name.begin(), Name.end());
    return;
  }
  
  // On systems that do not allow quoted names, we need to mangle most
  // strange characters.
  if (!MAI.doesAllowQuotesInName())
    return appendMangledName(OutName, Name, MAI);
  
  // Okay, the system allows quoted strings.  We can quote most anything, the
  // only characters that need escaping are " and \n.
  assert(Name.find_first_of("\n\"") != StringRef::npos);
  return appendMangledQuotedName(OutName, Name);
}

/// AddFastCallStdCallSuffix - Microsoft fastcall and stdcall functions require
/// a suffix on their name indicating the number of words of arguments they
/// take.
static void AddFastCallStdCallSuffix(SmallVectorImpl<char> &OutName,
                                     const Function *F, const DataLayout &TD) {
  // Calculate arguments size total.
  unsigned ArgWords = 0;
  for (Function::const_arg_iterator AI = F->arg_begin(), AE = F->arg_end();
       AI != AE; ++AI) {
    Type *Ty = AI->getType();
    // 'Dereference' type in case of byval parameter attribute
    if (AI->hasByValAttr())
      Ty = cast<PointerType>(Ty)->getElementType();
    // Size should be aligned to DWORD boundary
    ArgWords += ((TD.getTypeAllocSize(Ty) + 3)/4)*4;
  }
  
  raw_svector_ostream(OutName) << '@' << ArgWords;
}


/// getNameWithPrefix - Fill OutName with the name of the appropriate prefix
/// and the specified global variable's name.  If the global variable doesn't
/// have a name, this fills in a unique name for the global.
void Mangler::getNameWithPrefix(SmallVectorImpl<char> &OutName,
                                const GlobalValue *GV,
                                bool isImplicitlyPrivate) {
  ManglerPrefixTy PrefixTy = Mangler::Default;
  if (GV->hasPrivateLinkage() || isImplicitlyPrivate)
    PrefixTy = Mangler::Private;
  else if (GV->hasLinkerPrivateLinkage() || GV->hasLinkerPrivateWeakLinkage())
    PrefixTy = Mangler::LinkerPrivate;
  
  // If this global has a name, handle it simply.
  if (GV->hasName()) {
    getNameWithPrefix(OutName, GV->getName(), PrefixTy);
  } else {
    // Get the ID for the global, assigning a new one if we haven't got one
    // already.
    unsigned &ID = AnonGlobalIDs[GV];
    if (ID == 0) ID = NextAnonGlobalID++;
  
    // Must mangle the global into a unique ID.
    getNameWithPrefix(OutName, "__unnamed_" + Twine(ID), PrefixTy);
  }
  
  // If we are supposed to add a microsoft-style suffix for stdcall/fastcall,
  // add it.
  if (Context.getAsmInfo().hasMicrosoftFastStdCallMangling()) {
    if (const Function *F = dyn_cast<Function>(GV)) {
      CallingConv::ID CC = F->getCallingConv();
    
      // fastcall functions need to start with @.
      // FIXME: This logic seems unlikely to be right.
      if (CC == CallingConv::X86_FastCall) {
        if (OutName[0] == '_')
          OutName[0] = '@';
        else
          OutName.insert(OutName.begin(), '@');
      }
    
      // fastcall and stdcall functions usually need @42 at the end to specify
      // the argument info.
      FunctionType *FT = F->getFunctionType();
      if ((CC == CallingConv::X86_FastCall || CC == CallingConv::X86_StdCall) &&
          // "Pure" variadic functions do not receive @0 suffix.
          (!FT->isVarArg() || FT->getNumParams() == 0 ||
           (FT->getNumParams() == 1 && F->hasStructRetAttr())))
        AddFastCallStdCallSuffix(OutName, F, TD);
    }
  }
}

/// getSymbol - Return the MCSymbol for the specified global value.  This
/// symbol is the main label that is the address of the global.
MCSymbol *Mangler::getSymbol(const GlobalValue *GV) {
  SmallString<60> NameStr;
  getNameWithPrefix(NameStr, GV, false);
  return Context.GetOrCreateSymbol(NameStr.str());
}


