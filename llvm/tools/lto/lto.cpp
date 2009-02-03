//===-lto.cpp - LLVM Link Time Optimizer ----------------------------------===//
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

#include "llvm-c/lto.h"

#include "LTOModule.h"
#include "LTOCodeGenerator.h"


// holds most recent error string
// *** not thread safe ***
static std::string sLastErrorString;



//
// returns a printable string
//
extern const char* lto_get_version()
{
    return LTOCodeGenerator::getVersionString();
}

//
// returns the last error string or NULL if last operation was successful
//
const char* lto_get_error_message()
{
    return sLastErrorString.c_str();
}



//
// validates if a file is a loadable object file
//
bool lto_module_is_object_file(const char* path)
{
    return LTOModule::isBitcodeFile(path);
}


//
// validates if a file is a loadable object file compilable for requested target
//
bool lto_module_is_object_file_for_target(const char* path, 
                                            const char* target_triplet_prefix)
{
    return LTOModule::isBitcodeFileForTarget(path, target_triplet_prefix);
}


//
// validates if a buffer is a loadable object file
//
bool lto_module_is_object_file_in_memory(const void* mem, size_t length)
{
    return LTOModule::isBitcodeFile(mem, length);
}


//
// validates if a buffer is a loadable object file compilable for the target
//
bool lto_module_is_object_file_in_memory_for_target(const void* mem, 
                            size_t length, const char* target_triplet_prefix)
{
    return LTOModule::isBitcodeFileForTarget(mem, length, target_triplet_prefix);
}



//
// loads an object file from disk  
// returns NULL on error (check lto_get_error_message() for details)
//
lto_module_t lto_module_create(const char* path)
{
     return LTOModule::makeLTOModule(path, sLastErrorString);
}


//
// loads an object file from memory 
// returns NULL on error (check lto_get_error_message() for details)
//
lto_module_t lto_module_create_from_memory(const void* mem, size_t length)
{
     return LTOModule::makeLTOModule(mem, length, sLastErrorString);
}


//
// frees all memory for a module
// upon return the lto_module_t is no longer valid
//
void lto_module_dispose(lto_module_t mod)
{
    delete mod;
}


//
// returns triplet string which the object module was compiled under
//
const char* lto_module_get_target_triple(lto_module_t mod)
{
    return mod->getTargetTriple();
}


//
// returns the number of symbols in the object module
//
uint32_t lto_module_get_num_symbols(lto_module_t mod)
{
    return mod->getSymbolCount();
}

//
// returns the name of the ith symbol in the object module
//
const char* lto_module_get_symbol_name(lto_module_t mod, uint32_t index)
{
    return mod->getSymbolName(index);
}


//
// returns the attributes of the ith symbol in the object module
//
lto_symbol_attributes lto_module_get_symbol_attribute(lto_module_t mod, 
                                                            uint32_t index)
{
    return mod->getSymbolAttributes(index);
}





//
// instantiates a code generator
// returns NULL if there is an error
//
lto_code_gen_t lto_codegen_create()
{
     return new LTOCodeGenerator();
}



//
// frees all memory for a code generator
// upon return the lto_code_gen_t is no longer valid
//
void lto_codegen_dispose(lto_code_gen_t cg)
{
    delete cg;
}



//
// add an object module to the set of modules for which code will be generated
// returns true on error (check lto_get_error_message() for details)
//
bool lto_codegen_add_module(lto_code_gen_t cg, lto_module_t mod)
{
    return cg->addModule(mod, sLastErrorString);
}


//
// sets what if any format of debug info should be generated
// returns true on error (check lto_get_error_message() for details)
//
bool lto_codegen_set_debug_model(lto_code_gen_t cg, lto_debug_model debug)
{
    return cg->setDebugInfo(debug, sLastErrorString);
}


//
// sets what code model to generated
// returns true on error (check lto_get_error_message() for details)
//
bool lto_codegen_set_pic_model(lto_code_gen_t cg, lto_codegen_model model)
{
    return cg->setCodePICModel(model, sLastErrorString);
}

//
// adds to a list of all global symbols that must exist in the final
// generated code.  If a function is not listed there, it might be
// inlined into every usage and optimized away.
//
void lto_codegen_add_must_preserve_symbol(lto_code_gen_t cg, const char* symbol)
{
    cg->addMustPreserveSymbol(symbol);
}


//
// writes a new file at the specified path that contains the
// merged contents of all modules added so far.
// returns true on error (check lto_get_error_message() for details)
//
bool lto_codegen_write_merged_modules(lto_code_gen_t cg, const char* path)
{
   return cg->writeMergedModules(path, sLastErrorString);
}


//
// Generates code for all added modules into one native object file.
// On sucess returns a pointer to a generated mach-o/ELF buffer and
// length set to the buffer size.  The buffer is owned by the 
// lto_code_gen_t and will be freed when lto_codegen_dispose()
// is called, or lto_codegen_compile() is called again.
// On failure, returns NULL (check lto_get_error_message() for details).
//
extern const void*
lto_codegen_compile(lto_code_gen_t cg, size_t* length)
{
    return cg->compile(length, sLastErrorString);
}


//
// Used to pass extra options to the code generator
//
extern void
lto_codegen_debug_options(lto_code_gen_t cg, const char * opt)
{
  cg->setCodeGenDebugOptions(opt);
}



