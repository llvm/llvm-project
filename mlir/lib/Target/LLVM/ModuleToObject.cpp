//===- ModuleToObject.cpp - Module to object base class ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the base class for transforming Operations into binary
// objects.
//
//===----------------------------------------------------------------------===//

#include "mlir/Target/LLVM/ModuleToObject.h"

#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"

#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/IPO/Internalize.h"

using namespace mlir;
using namespace mlir::LLVM;

ModuleToObject::ModuleToObject(Operation &module, StringRef triple,
                               StringRef chip, StringRef features, int optLevel)
    : module(module), triple(triple), chip(chip), features(features),
      optLevel(optLevel) {}

Operation &ModuleToObject::getOperation() { return module; }

std::unique_ptr<llvm::TargetMachine> ModuleToObject::createTargetMachine() {
  std::string error;
  // Load the target.
  const llvm::Target *target =
      llvm::TargetRegistry::lookupTarget(triple, error);
  if (!target) {
    getOperation().emitError() << "Failed to lookup target: " << error;
    return {};
  }

  // Create the target machine using the target.
  llvm::TargetMachine *machine =
      target->createTargetMachine(triple, chip, features, {}, {});
  if (!machine) {
    getOperation().emitError() << "Failed to create the target machine.";
    return {};
  }
  return std::unique_ptr<llvm::TargetMachine>{machine};
}

std::unique_ptr<llvm::Module>
ModuleToObject::loadBitcodeFile(llvm::LLVMContext &context,
                                llvm::TargetMachine &targetMachine,
                                StringRef path) {
  llvm::SMDiagnostic error;
  std::unique_ptr<llvm::Module> library =
      llvm::getLazyIRFileModule(path, error, context);
  if (!library) {
    getOperation().emitError() << "Failed loading file from " << path
                               << ", error: " << error.getMessage();
    return nullptr;
  }
  if (failed(handleBitcodeFile(*library, targetMachine))) {
    return nullptr;
  }
  return library;
}

LogicalResult ModuleToObject::loadBitcodeFilesFromList(
    llvm::LLVMContext &context, llvm::TargetMachine &targetMachine,
    ArrayRef<std::string> fileList,
    SmallVector<std::unique_ptr<llvm::Module>> &llvmModules,
    bool failureOnError) {
  for (const std::string &str : fileList) {
    // Test if the path exists, if it doesn't abort.
    StringRef pathRef = StringRef(str.data(), str.size());
    if (!llvm::sys::fs::is_regular_file(pathRef)) {
      getOperation().emitError()
          << "File path: " << pathRef << " does not exist or is not a file.\n";
      return failure();
    }
    // Load the file or abort on error.
    if (auto bcFile = loadBitcodeFile(context, targetMachine, pathRef))
      llvmModules.push_back(std::move(bcFile));
    else if (failureOnError)
      return failure();
  }
  return success();
}

std::unique_ptr<llvm::Module>
ModuleToObject::translateToLLVMIR(llvm::LLVMContext &llvmContext) {
  return translateModuleToLLVMIR(&getOperation(), llvmContext);
}

LogicalResult
ModuleToObject::linkFiles(llvm::Module &module,
                          SmallVector<std::unique_ptr<llvm::Module>> &&libs) {
  if (libs.empty())
    return success();
  llvm::Linker linker(module);
  for (std::unique_ptr<llvm::Module> &libModule : libs) {
    // This bitcode linking imports the library functions into the module,
    // allowing LLVM optimization passes (which must run after linking) to
    // optimize across the libraries and the module's code. We also only import
    // symbols if they are referenced by the module or a previous library since
    // there will be no other source of references to those symbols in this
    // compilation and since we don't want to bloat the resulting code object.
    bool err = linker.linkInModule(
        std::move(libModule), llvm::Linker::Flags::LinkOnlyNeeded,
        [](llvm::Module &m, const StringSet<> &gvs) {
          llvm::internalizeModule(m, [&gvs](const llvm::GlobalValue &gv) {
            return !gv.hasName() || (gvs.count(gv.getName()) == 0);
          });
        });
    // True is linker failure
    if (err) {
      getOperation().emitError("Unrecoverable failure during bitcode linking.");
      // We have no guaranties about the state of `ret`, so bail
      return failure();
    }
  }
  return success();
}

LogicalResult ModuleToObject::optimizeModule(llvm::Module &module,
                                             llvm::TargetMachine &targetMachine,
                                             int optLevel) {
  if (optLevel < 0 || optLevel > 3)
    return getOperation().emitError()
           << "Invalid optimization level: " << optLevel << ".";

  targetMachine.setOptLevel(static_cast<llvm::CodeGenOpt::Level>(optLevel));

  auto transformer =
      makeOptimizingTransformer(optLevel, /*sizeLevel=*/0, &targetMachine);
  auto error = transformer(&module);
  if (error) {
    InFlightDiagnostic mlirError = getOperation().emitError();
    llvm::handleAllErrors(
        std::move(error), [&mlirError](const llvm::ErrorInfoBase &ei) {
          mlirError << "Could not optimize LLVM IR: " << ei.message() << "\n";
        });
    return mlirError;
  }
  return success();
}

std::optional<std::string>
ModuleToObject::translateToISA(llvm::Module &llvmModule,
                               llvm::TargetMachine &targetMachine) {
  std::string targetISA;
  llvm::raw_string_ostream stream(targetISA);

  { // Drop pstream after this to prevent the ISA from being stuck buffering
    llvm::buffer_ostream pstream(stream);
    llvm::legacy::PassManager codegenPasses;

    if (targetMachine.addPassesToEmitFile(codegenPasses, pstream, nullptr,
                                          llvm::CGFT_AssemblyFile))
      return std::nullopt;

    codegenPasses.run(llvmModule);
  }
  return stream.str();
}

std::optional<SmallVector<char, 0>>
ModuleToObject::moduleToObject(llvm::Module &llvmModule,
                               llvm::TargetMachine &targetMachine) {
  SmallVector<char, 0> binaryData;
  // Write the LLVM module bitcode to a buffer.
  llvm::raw_svector_ostream outputStream(binaryData);
  llvm::WriteBitcodeToFile(llvmModule, outputStream);
  return binaryData;
}

std::optional<SmallVector<char, 0>> ModuleToObject::run() {
  // Translate the module to LLVM IR.
  llvm::LLVMContext llvmContext;
  std::unique_ptr<llvm::Module> llvmModule = translateToLLVMIR(llvmContext);
  if (!llvmModule) {
    getOperation().emitError() << "Failed creating the llvm::Module.";
    return std::nullopt;
  }

  // Create the target machine.
  std::unique_ptr<llvm::TargetMachine> targetMachine = createTargetMachine();
  if (!targetMachine)
    return std::nullopt;

  // Set the data layout and target triple of the module.
  llvmModule->setDataLayout(targetMachine->createDataLayout());
  llvmModule->setTargetTriple(targetMachine->getTargetTriple().getTriple());

  // Link bitcode files.
  handleModulePreLink(*llvmModule, *targetMachine);
  {
    auto libs = loadBitcodeFiles(*llvmModule, *targetMachine);
    if (!libs)
      return std::nullopt;
    if (!libs->empty())
      if (failed(linkFiles(*llvmModule, std::move(*libs))))
        return std::nullopt;
    handleModulePostLink(*llvmModule, *targetMachine);
  }

  // Optimize the module.
  if (failed(optimizeModule(*llvmModule, *targetMachine, optLevel)))
    return std::nullopt;

  // Return the serialized object.
  return moduleToObject(*llvmModule, *targetMachine);
}
