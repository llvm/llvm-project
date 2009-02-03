//===-LTOCodeGenerator.h - LLVM Link Time Optimizer -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file declares the LTOCodeGenerator class. 
//
//===----------------------------------------------------------------------===//


#ifndef LTO_CODE_GENERATOR_H
#define LTO_CODE_GENERATOR_H

#include "llvm/Linker.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/SmallVector.h"

#include <string>


//
// C++ class which implements the opaque lto_code_gen_t
//

class LTOCodeGenerator {
public:
    static const char*        getVersionString();
    
                            LTOCodeGenerator();
                            ~LTOCodeGenerator();
                            
    bool                addModule(class LTOModule*, std::string& errMsg);
    bool                setDebugInfo(lto_debug_model, std::string& errMsg);
    bool                setCodePICModel(lto_codegen_model, std::string& errMsg);
    void                addMustPreserveSymbol(const char* sym);
    bool                writeMergedModules(const char* path, 
                                                           std::string& errMsg);
    const void*         compile(size_t* length, std::string& errMsg);
    void                setCodeGenDebugOptions(const char *opts); 
private:
    bool                generateAssemblyCode(llvm::raw_ostream& out, 
                                             std::string& errMsg);
    bool                assemble(const std::string& asmPath, 
                            const std::string& objPath, std::string& errMsg);
    void                applyScopeRestrictions();
    bool                determineTarget(std::string& errMsg);
    
    typedef llvm::StringMap<uint8_t> StringSet;

    llvm::Linker                _linker;
    llvm::TargetMachine*        _target;
    bool                        _emitDwarfDebugInfo;
    bool                        _scopeRestrictionsDone;
    lto_codegen_model           _codeModel;
    StringSet                   _mustPreserveSymbols;
    llvm::MemoryBuffer*         _nativeObjectFile;
    std::vector<const char*>    _codegenOptions;
};

#endif // LTO_CODE_GENERATOR_H

