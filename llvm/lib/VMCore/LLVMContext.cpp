//===-- LLVMContext.cpp - Implement LLVMContext -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements LLVMContext, as a wrapper around the opaque
//  class LLVMContextImpl.
//
//===----------------------------------------------------------------------===//

#include "llvm/LLVMContext.h"
#include "llvm/Metadata.h"
#include "llvm/Constants.h"
#include "llvm/Instruction.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/SourceMgr.h"
#include "LLVMContextImpl.h"
#include <cctype>
using namespace llvm;

static ManagedStatic<LLVMContext> GlobalContext;

LLVMContext& llvm::getGlobalContext() {
  return *GlobalContext;
}

LLVMContext::LLVMContext() : pImpl(new LLVMContextImpl(*this)) {
  // Create the fixed metadata kinds. This is done in the same order as the
  // MD_* enum values so that they correspond.

  // Create the 'dbg' metadata kind.
  unsigned DbgID = getMDKindID("dbg");
  assert(DbgID == MD_dbg && "dbg kind id drifted"); (void)DbgID;

  // Create the 'tbaa' metadata kind.
  unsigned TBAAID = getMDKindID("tbaa");
  assert(TBAAID == MD_tbaa && "tbaa kind id drifted"); (void)TBAAID;

  // Create the 'prof' metadata kind.
  unsigned ProfID = getMDKindID("prof");
  assert(ProfID == MD_prof && "prof kind id drifted"); (void)ProfID;

  // Create the 'fpmath' metadata kind.
  unsigned FPAccuracyID = getMDKindID("fpmath");
  assert(FPAccuracyID == MD_fpmath && "fpmath kind id drifted");
  (void)FPAccuracyID;

  // Create the 'range' metadata kind.
  unsigned RangeID = getMDKindID("range");
  assert(RangeID == MD_range && "range kind id drifted");
  (void)RangeID;

  // Create the 'tbaa.struct' metadata kind.
  unsigned TBAAStructID = getMDKindID("tbaa.struct");
  assert(TBAAStructID == MD_tbaa_struct && "tbaa.struct kind id drifted");
  (void)TBAAStructID;
}
LLVMContext::~LLVMContext() { delete pImpl; }

void LLVMContext::addModule(Module *M) {
  pImpl->OwnedModules.insert(M);
}

void LLVMContext::removeModule(Module *M) {
  pImpl->OwnedModules.erase(M);
}

//===----------------------------------------------------------------------===//
// Recoverable Backend Errors
//===----------------------------------------------------------------------===//

void LLVMContext::
setInlineAsmDiagnosticHandler(InlineAsmDiagHandlerTy DiagHandler,
                              void *DiagContext) {
  pImpl->InlineAsmDiagHandler = DiagHandler;
  pImpl->InlineAsmDiagContext = DiagContext;
}

/// getInlineAsmDiagnosticHandler - Return the diagnostic handler set by
/// setInlineAsmDiagnosticHandler.
LLVMContext::InlineAsmDiagHandlerTy
LLVMContext::getInlineAsmDiagnosticHandler() const {
  return pImpl->InlineAsmDiagHandler;
}

/// getInlineAsmDiagnosticContext - Return the diagnostic context set by
/// setInlineAsmDiagnosticHandler.
void *LLVMContext::getInlineAsmDiagnosticContext() const {
  return pImpl->InlineAsmDiagContext;
}

void LLVMContext::emitError(const Twine &ErrorStr) {
  emitError(0U, ErrorStr);
}

void LLVMContext::emitError(const Instruction *I, const Twine &ErrorStr) {
  unsigned LocCookie = 0;
  if (const MDNode *SrcLoc = I->getMetadata("srcloc")) {
    if (SrcLoc->getNumOperands() != 0)
      if (const ConstantInt *CI = dyn_cast<ConstantInt>(SrcLoc->getOperand(0)))
        LocCookie = CI->getZExtValue();
  }
  return emitError(LocCookie, ErrorStr);
}

void LLVMContext::emitError(unsigned LocCookie, const Twine &ErrorStr) {
  // If there is no error handler installed, just print the error and exit.
  if (pImpl->InlineAsmDiagHandler == 0) {
    errs() << "error: " << ErrorStr << "\n";
    exit(1);
  }

  // If we do have an error handler, we can report the error and keep going.
  SMDiagnostic Diag("", SourceMgr::DK_Error, ErrorStr.str());

  pImpl->InlineAsmDiagHandler(Diag, pImpl->InlineAsmDiagContext, LocCookie);
}

//===----------------------------------------------------------------------===//
// Metadata Kind Uniquing
//===----------------------------------------------------------------------===//

#ifndef NDEBUG
/// isValidName - Return true if Name is a valid custom metadata handler name.
static bool isValidName(StringRef MDName) {
  if (MDName.empty())
    return false;

  if (!std::isalpha(MDName[0]))
    return false;

  for (StringRef::iterator I = MDName.begin() + 1, E = MDName.end(); I != E;
       ++I) {
    if (!std::isalnum(*I) && *I != '_' && *I != '-' && *I != '.')
      return false;
  }
  return true;
}
#endif

/// getMDKindID - Return a unique non-zero ID for the specified metadata kind.
unsigned LLVMContext::getMDKindID(StringRef Name) const {
  assert(isValidName(Name) && "Invalid MDNode name");

  // If this is new, assign it its ID.
  return
    pImpl->CustomMDKindNames.GetOrCreateValue(
      Name, pImpl->CustomMDKindNames.size()).second;
}

/// getHandlerNames - Populate client supplied smallvector using custome
/// metadata name and ID.
void LLVMContext::getMDKindNames(SmallVectorImpl<StringRef> &Names) const {
  Names.resize(pImpl->CustomMDKindNames.size());
  for (StringMap<unsigned>::const_iterator I = pImpl->CustomMDKindNames.begin(),
       E = pImpl->CustomMDKindNames.end(); I != E; ++I)
    Names[I->second] = I->first();
}
