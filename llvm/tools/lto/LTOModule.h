//===-LTOModule.h - LLVM Link Time Optimizer ------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the LTOModule class.
//
//===----------------------------------------------------------------------===//

#ifndef LTO_MODULE_H
#define LTO_MODULE_H

#include "llvm/Module.h"
#include "llvm/MC/MCContext.h"
#include "llvm/Target/Mangler.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/StringMap.h"
#include "llvm-c/lto.h"
#include <vector>
#include <string>

// Forward references to llvm classes.
namespace llvm {
  class Function;
  class GlobalValue;
  class MemoryBuffer;
  class TargetOptions;
  class Value;
}

//===----------------------------------------------------------------------===//
/// LTOModule - C++ class which implements the opaque lto_module_t type.
///
struct LTOModule {
private:
  typedef llvm::StringMap<uint8_t> StringSet;

  struct NameAndAttributes {
    const char        *name;
    uint32_t           attributes;
    bool               isFunction;
    llvm::GlobalValue *symbol;
  };

  llvm::OwningPtr<llvm::Module>           _module;
  llvm::OwningPtr<llvm::TargetMachine>    _target;
  std::vector<NameAndAttributes>          _symbols;

  // _defines and _undefines only needed to disambiguate tentative definitions
  StringSet                               _defines;
  llvm::StringMap<NameAndAttributes>      _undefines;
  std::vector<const char*>                _asm_undefines;
  llvm::MCContext                         _context;

  // Use mangler to add GlobalPrefix to names to match linker names.
  llvm::Mangler                           _mangler;

  LTOModule(llvm::Module *m, llvm::TargetMachine *t);
public:
  /// isBitcodeFile - Returns 'true' if the file or memory contents is LLVM
  /// bitcode.
  static bool isBitcodeFile(const void *mem, size_t length);
  static bool isBitcodeFile(const char *path);

  /// isBitcodeFileForTarget - Returns 'true' if the file or memory contents
  /// is LLVM bitcode for the specified triple.
  static bool isBitcodeFileForTarget(const void *mem,
                                     size_t length,
                                     const char *triplePrefix);
  static bool isBitcodeFileForTarget(const char *path,
                                     const char *triplePrefix);

  /// makeLTOModule - Create an LTOModule. N.B. These methods take ownership
  /// of the buffer.
  static LTOModule *makeLTOModule(const char* path,
                                  std::string &errMsg);
  static LTOModule *makeLTOModule(int fd, const char *path,
                                  size_t size, std::string &errMsg);
  static LTOModule *makeLTOModule(int fd, const char *path,
                                  size_t file_size,
                                  size_t map_size,
                                  off_t offset,
                                  std::string& errMsg);
  static LTOModule *makeLTOModule(const void *mem, size_t length,
                                  std::string &errMsg);

  /// getTargetTriple - Return the Module's target triple.
  const char *getTargetTriple() {
    return _module->getTargetTriple().c_str();
  }

  /// setTargetTriple - Set the Module's target triple.
  void setTargetTriple(const char *triple) {
    _module->setTargetTriple(triple);
  }

  /// getSymbolCount - Get the number of symbols
  uint32_t getSymbolCount() {
    return _symbols.size();
  }

  /// getSymbolAttributes - Get the attributes for a symbol at the specified
  /// index.
  lto_symbol_attributes getSymbolAttributes(uint32_t index) {
    if (index < _symbols.size())
      return lto_symbol_attributes(_symbols[index].attributes);
    return lto_symbol_attributes(0);
  }

  /// getSymbolName - Get the name of the symbol at the specified index.
  const char *getSymbolName(uint32_t index) {
    if (index < _symbols.size())
      return _symbols[index].name;
    return NULL;
  }

  /// getLLVVMModule - Return the Module.
  llvm::Module *getLLVVMModule() { return _module.get(); }

  /// getAsmUndefinedRefs -
  const std::vector<const char*> &getAsmUndefinedRefs() {
    return _asm_undefines;
  }

  /// getTargetOptions - Fill the TargetOptions object with the options
  /// specified on the command line.
  static void getTargetOptions(llvm::TargetOptions &Options);

private:
  /// parseSymbols - Parse the symbols from the module and model-level ASM and
  /// add them to either the defined or undefined lists.
  bool parseSymbols(std::string &errMsg);

  /// addPotentialUndefinedSymbol - Add a symbol which isn't defined just yet
  /// to a list to be resolved later.
  void addPotentialUndefinedSymbol(llvm::GlobalValue *dcl, bool isFunc);

  /// addDefinedSymbol - Add a defined symbol to the list.
  void addDefinedSymbol(llvm::GlobalValue *def, bool isFunction);

  /// addDefinedFunctionSymbol - Add a function symbol as defined to the list.
  void addDefinedFunctionSymbol(llvm::Function *f);

  /// addDefinedDataSymbol - Add a data symbol as defined to the list.
  void addDefinedDataSymbol(llvm::GlobalValue *v);

  /// addAsmGlobalSymbols - Add global symbols from module-level ASM to the
  /// defined or undefined lists.
  bool addAsmGlobalSymbols(std::string &errMsg);

  /// addAsmGlobalSymbol - Add a global symbol from module-level ASM to the
  /// defined list.
  void addAsmGlobalSymbol(const char *, lto_symbol_attributes scope);

  /// addAsmGlobalSymbolUndef - Add a global symbol from module-level ASM to
  /// the undefined list.
  void addAsmGlobalSymbolUndef(const char *);

  /// addObjCClass - Parse i386/ppc ObjC class data structure.
  void addObjCClass(llvm::GlobalVariable *clgv);

  /// addObjCCategory - Parse i386/ppc ObjC category data structure.
  void addObjCCategory(llvm::GlobalVariable *clgv);

  /// addObjCClassRef - Parse i386/ppc ObjC class list data structure.
  void addObjCClassRef(llvm::GlobalVariable *clgv);

  /// objcClassNameFromExpression - Get string that the data pointer points
  /// to.
  bool objcClassNameFromExpression(llvm::Constant* c, std::string &name);

  /// isTargetMatch - Returns 'true' if the memory buffer is for the specified
  /// target triple.
  static bool isTargetMatch(llvm::MemoryBuffer *memBuffer,
                            const char *triplePrefix);

  /// makeLTOModule - Create an LTOModule (private version). N.B. This
  /// method takes ownership of the buffer.
  static LTOModule *makeLTOModule(llvm::MemoryBuffer *buffer,
                                  std::string &errMsg);

  /// makeBuffer - Create a MemoryBuffer from a memory range.
  static llvm::MemoryBuffer *makeBuffer(const void *mem, size_t length);
};

#endif // LTO_MODULE_H
