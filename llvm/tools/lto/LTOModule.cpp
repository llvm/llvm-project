//===-LTOModule.cpp - LLVM Link Time Optimizer ----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file implements the Link Time Optimization library. This library is 
// intended to be used by linker to optimize code at link time.
//
//===----------------------------------------------------------------------===//

#include "LTOModule.h"

#include "llvm/Module.h"
#include "llvm/ModuleProvider.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/Support/SystemUtils.h"
#include "llvm/Support/Mangler.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/System/Path.h"
#include "llvm/System/Process.h"
#include "llvm/Target/SubtargetFeature.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetMachineRegistry.h"
#include "llvm/Target/TargetAsmInfo.h"

#include <fstream>

using namespace llvm;

bool LTOModule::isBitcodeFile(const void* mem, size_t length)
{
    return ( llvm::sys::IdentifyFileType((char*)mem, length) 
                                            == llvm::sys::Bitcode_FileType );
}

bool LTOModule::isBitcodeFile(const char* path)
{
    return llvm::sys::Path(path).isBitcodeFile();
}

bool LTOModule::isBitcodeFileForTarget(const void* mem, size_t length,
                                       const char* triplePrefix) 
{
    MemoryBuffer* buffer = makeBuffer(mem, length);
    if ( buffer == NULL )
        return false;
    return isTargetMatch(buffer, triplePrefix);
}


bool LTOModule::isBitcodeFileForTarget(const char* path,
                                       const char* triplePrefix) 
{
    MemoryBuffer *buffer = MemoryBuffer::getFile(path);
    if (buffer == NULL)
        return false;
    return isTargetMatch(buffer, triplePrefix);
}

// takes ownership of buffer
bool LTOModule::isTargetMatch(MemoryBuffer* buffer, const char* triplePrefix)
{
    OwningPtr<ModuleProvider> mp(getBitcodeModuleProvider(buffer));
    // on success, mp owns buffer and both are deleted at end of this method
    if ( !mp ) {
        delete buffer;
        return false;
    }
    std::string actualTarget = mp->getModule()->getTargetTriple();
    return ( strncmp(actualTarget.c_str(), triplePrefix, 
                    strlen(triplePrefix)) == 0);
}


LTOModule::LTOModule(Module* m, TargetMachine* t) 
 : _module(m), _target(t), _symbolsParsed(false)
{
}

LTOModule* LTOModule::makeLTOModule(const char* path, std::string& errMsg)
{
    OwningPtr<MemoryBuffer> buffer(MemoryBuffer::getFile(path, &errMsg));
    if ( !buffer )
        return NULL;
    return makeLTOModule(buffer.get(), errMsg);
}

/// makeBuffer - create a MemoryBuffer from a memory range.
/// MemoryBuffer requires the byte past end of the buffer to be a zero.
/// We might get lucky and already be that way, otherwise make a copy.
/// Also if next byte is on a different page, don't assume it is readable.
MemoryBuffer* LTOModule::makeBuffer(const void* mem, size_t length)
{
    const char* startPtr = (char*)mem;
    const char* endPtr = startPtr+length;
    if ( (((uintptr_t)endPtr & (sys::Process::GetPageSize()-1)) == 0) 
        || (*endPtr != 0) ) 
        return MemoryBuffer::getMemBufferCopy(startPtr, endPtr);
    else
        return MemoryBuffer::getMemBuffer(startPtr, endPtr);
}


LTOModule* LTOModule::makeLTOModule(const void* mem, size_t length, 
                                                        std::string& errMsg)
{
    OwningPtr<MemoryBuffer> buffer(makeBuffer(mem, length));
    if ( !buffer )
        return NULL;
    return makeLTOModule(buffer.get(), errMsg);
}

/// getFeatureString - Return a string listing the features associated with the
/// target triple.
///
/// FIXME: This is an inelegant way of specifying the features of a
/// subtarget. It would be better if we could encode this information into the
/// IR. See <rdar://5972456>.
std::string getFeatureString(const char *TargetTriple) {
  SubtargetFeatures Features;

  if (strncmp(TargetTriple, "powerpc-apple-", 14) == 0) {
    Features.AddFeature("altivec", true);
  } else if (strncmp(TargetTriple, "powerpc64-apple-", 16) == 0) {
    Features.AddFeature("64bit", true);
    Features.AddFeature("altivec", true);
  }

  return Features.getString();
}

LTOModule* LTOModule::makeLTOModule(MemoryBuffer* buffer, std::string& errMsg)
{
    // parse bitcode buffer
    OwningPtr<Module> m(ParseBitcodeFile(buffer, &errMsg));
    if ( !m )
        return NULL;
    // find machine architecture for this module
    const TargetMachineRegistry::entry* march = 
            TargetMachineRegistry::getClosestStaticTargetForModule(*m, errMsg);

    if ( march == NULL ) 
        return NULL;

    // construct LTModule, hand over ownership of module and target
    std::string FeatureStr = getFeatureString(m->getTargetTriple().c_str());
    TargetMachine* target = march->CtorFn(*m, FeatureStr);
    return new LTOModule(m.take(), target);
}


const char* LTOModule::getTargetTriple()
{
    return _module->getTargetTriple().c_str();
}

void LTOModule::addDefinedFunctionSymbol(Function* f, Mangler &mangler)
{
    // add to list of defined symbols
    addDefinedSymbol(f, mangler, true); 

    // add external symbols referenced by this function.
    for (Function::iterator b = f->begin(); b != f->end(); ++b) {
        for (BasicBlock::iterator i = b->begin(); i != b->end(); ++i) {
            for (unsigned count = 0, total = i->getNumOperands(); 
                                        count != total; ++count) {
                findExternalRefs(i->getOperand(count), mangler);
            }
        }
    }
}

void LTOModule::addDefinedDataSymbol(GlobalValue* v, Mangler &mangler)
{    
    // add to list of defined symbols
    addDefinedSymbol(v, mangler, false); 

    // add external symbols referenced by this data.
    for (unsigned count = 0, total = v->getNumOperands();
                                                count != total; ++count) {
        findExternalRefs(v->getOperand(count), mangler);
    }
}


void LTOModule::addDefinedSymbol(GlobalValue* def, Mangler &mangler, 
                                bool isFunction)
{    
    // string is owned by _defines
    const char* symbolName = ::strdup(mangler.getValueName(def).c_str());
    
    // set alignment part log2() can have rounding errors
    uint32_t align = def->getAlignment();
    uint32_t attr = align ? CountTrailingZeros_32(def->getAlignment()) : 0;
    
    // set permissions part
    if ( isFunction )
        attr |= LTO_SYMBOL_PERMISSIONS_CODE;
    else {
        GlobalVariable* gv = dyn_cast<GlobalVariable>(def);
        if ( (gv != NULL) && gv->isConstant() )
            attr |= LTO_SYMBOL_PERMISSIONS_RODATA;
        else
            attr |= LTO_SYMBOL_PERMISSIONS_DATA;
    }
    
    // set definition part 
    if ( def->hasWeakLinkage() || def->hasLinkOnceLinkage() ) {
        attr |= LTO_SYMBOL_DEFINITION_WEAK;
    }
    else if ( def->hasCommonLinkage()) {
        attr |= LTO_SYMBOL_DEFINITION_TENTATIVE;
    }
    else { 
        attr |= LTO_SYMBOL_DEFINITION_REGULAR;
    }
    
    // set scope part
    if ( def->hasHiddenVisibility() )
        attr |= LTO_SYMBOL_SCOPE_HIDDEN;
    else if ( def->hasProtectedVisibility() )
        attr |= LTO_SYMBOL_SCOPE_PROTECTED;
    else if ( def->hasExternalLinkage() || def->hasWeakLinkage() 
              || def->hasLinkOnceLinkage() || def->hasCommonLinkage() )
        attr |= LTO_SYMBOL_SCOPE_DEFAULT;
    else
        attr |= LTO_SYMBOL_SCOPE_INTERNAL;

    // add to table of symbols
    NameAndAttributes info;
    info.name = symbolName;
    info.attributes = (lto_symbol_attributes)attr;
    _symbols.push_back(info);
    _defines[info.name] = 1;
}

void LTOModule::addAsmGlobalSymbol(const char *name) {
  // string is owned by _defines
  const char *symbolName = ::strdup(name);
  uint32_t attr = LTO_SYMBOL_DEFINITION_REGULAR;
  attr |= LTO_SYMBOL_SCOPE_DEFAULT;

  // add to table of symbols
  NameAndAttributes info;
  info.name = symbolName;
  info.attributes = (lto_symbol_attributes)attr;
  _symbols.push_back(info);
  _defines[info.name] = 1;
}

void LTOModule::addPotentialUndefinedSymbol(GlobalValue* decl, Mangler &mangler)
{   
   const char* name = mangler.getValueName(decl).c_str();
    // ignore all llvm.* symbols
    if ( strncmp(name, "llvm.", 5) != 0 ) {
        _undefines[name] = 1;
    }
}



// Find exeternal symbols referenced by VALUE. This is a recursive function.
void LTOModule::findExternalRefs(Value* value, Mangler &mangler) {

    if (GlobalValue* gv = dyn_cast<GlobalValue>(value)) {
        if ( !gv->hasExternalLinkage() )
            addPotentialUndefinedSymbol(gv, mangler);
        // If this is a variable definition, do not recursively process
        // initializer.  It might contain a reference to this variable
        // and cause an infinite loop.  The initializer will be
        // processed in addDefinedDataSymbol(). 
        return;
    }
    
    // GlobalValue, even with InternalLinkage type, may have operands with 
    // ExternalLinkage type. Do not ignore these operands.
    if (Constant* c = dyn_cast<Constant>(value)) {
        // Handle ConstantExpr, ConstantStruct, ConstantArry etc..
        for (unsigned i = 0, e = c->getNumOperands(); i != e; ++i)
            findExternalRefs(c->getOperand(i), mangler);
    }
}

void LTOModule::lazyParseSymbols()
{
    if ( !_symbolsParsed ) {
        _symbolsParsed = true;
        
        // Use mangler to add GlobalPrefix to names to match linker names.
        Mangler mangler(*_module, _target->getTargetAsmInfo()->getGlobalPrefix());

        // add functions
        for (Module::iterator f = _module->begin(); f != _module->end(); ++f) {
            if ( f->isDeclaration() ) 
                addPotentialUndefinedSymbol(f, mangler);
            else 
                addDefinedFunctionSymbol(f, mangler);
        }
        
        // add data 
        for (Module::global_iterator v = _module->global_begin(), 
                                    e = _module->global_end(); v !=  e; ++v) {
            if ( v->isDeclaration() ) 
                addPotentialUndefinedSymbol(v, mangler);
            else 
                addDefinedDataSymbol(v, mangler);
        }

        // add asm globals
        const std::string &inlineAsm = _module->getModuleInlineAsm();
        const std::string glbl = ".globl";
        std::string asmSymbolName;
        std::string::size_type pos = inlineAsm.find(glbl, 0);
        while (pos != std::string::npos) {
          // eat .globl
          pos = pos + 6;

          // skip white space between .globl and symbol name
          std::string::size_type pbegin = inlineAsm.find_first_not_of(' ', pos);
          if (pbegin == std::string::npos)
            break;

          // find end-of-line
          std::string::size_type pend = inlineAsm.find_first_of('\n', pbegin);
          if (pend == std::string::npos)
            break;

          asmSymbolName.assign(inlineAsm, pbegin, pend - pbegin);
          addAsmGlobalSymbol(asmSymbolName.c_str());

          // search next .globl
          pos = inlineAsm.find(glbl, pend);
        }

        // make symbols for all undefines
        for (StringSet::iterator it=_undefines.begin(); 
                                                it != _undefines.end(); ++it) {
            // if this symbol also has a definition, then don't make an undefine
            // because it is a tentative definition
            if ( _defines.count(it->getKeyData(), it->getKeyData()+
                                                  it->getKeyLength()) == 0 ) {
                NameAndAttributes info;
                info.name = it->getKeyData();
                info.attributes = LTO_SYMBOL_DEFINITION_UNDEFINED;
                _symbols.push_back(info);
            }
        }
    }    
}


uint32_t LTOModule::getSymbolCount()
{
    lazyParseSymbols();
    return _symbols.size();
}


lto_symbol_attributes LTOModule::getSymbolAttributes(uint32_t index)
{
    lazyParseSymbols();
    if ( index < _symbols.size() )
        return _symbols[index].attributes;
    else
        return lto_symbol_attributes(0);
}

const char* LTOModule::getSymbolName(uint32_t index)
{
    lazyParseSymbols();
    if ( index < _symbols.size() )
        return _symbols[index].name;
    else
        return NULL;
}

