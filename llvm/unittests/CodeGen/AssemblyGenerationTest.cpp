#include "gtest/gtest.h"


#include "llvm/IR/Module.h"
#include "llvm/IR/IRBuilder.h"
#include <iostream>
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/IR/LLVMContext.h"


TEST(AsmPrinter, EmitAssemblyFile) {
    llvm::LLVMContext context;
    llvm::Module module("MyModule", context);

    // Create a simple "main" function returning 1
    llvm::IRBuilder<> builder(context);
    llvm::FunctionType *functype = llvm::FunctionType::get(builder.getInt32Ty(), false);
    llvm::Function *function = llvm::Function::Create(
        functype, llvm::Function::ExternalLinkage, "main", module);
    llvm::BasicBlock *EntryBlock = llvm::BasicBlock::Create(context, "entry", function);
    builder.SetInsertPoint(EntryBlock);
    builder.CreateRet(builder.getInt32(1));

    // Verify module is valid
    EXPECT_FALSE(llvm::verifyModule(module, &llvm::errs())) << "Module verification failed";

    // Initialize targets
    llvm::InitializeAllTargetInfos();
    llvm::InitializeAllTargets();
    llvm::InitializeAllTargetMCs();
    llvm::InitializeAllAsmParsers();
    
    llvm::InitializeAllAsmPrinters();

    std::string Error;
    // auto TargetTriple = llvm::sys::getDefaultTargetTriple();
    // Set target triple on the module
    auto triple = llvm::Triple(llvm::sys::getDefaultTargetTriple());
    auto target = llvm::TargetRegistry::lookupTarget(triple, Error);

    module.setTargetTriple(triple);

    auto CPU = "generic";
    auto Features = "";
    llvm::TargetOptions opt;
    auto TargetMachine = target->createTargetMachine(triple, CPU, Features, opt, llvm::Reloc::PIC_);
    ASSERT_NE(TargetMachine, nullptr);

    module.setDataLayout(TargetMachine->createDataLayout());

    // Open file to write assembly
    std::error_code EC;
    llvm::raw_fd_ostream dest("file.s", EC, llvm::sys::fs::OF_None);
    ASSERT_FALSE(EC) << "Could not open file: " << EC.message();

    llvm::legacy::PassManager pass;
    auto FileType = llvm::CodeGenFileType::AssemblyFile;

    ASSERT_FALSE(TargetMachine->addPassesToEmitFile(pass, dest, nullptr, FileType))
        << "TargetMachine can't emit a file of this type";

    // Run the pass
    pass.run(module);
    dest.flush();

    // Optionally: check that the file exists
    EXPECT_TRUE(llvm::sys::fs::exists("file.s"));
}
