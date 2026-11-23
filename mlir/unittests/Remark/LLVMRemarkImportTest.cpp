//===- LLVMRemarkImportTest.cpp - LLVM remark import unit tests -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Remark/LLVMRemarkImport.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Remarks.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Parser/Parser.h"

#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Remarks/RemarkFormat.h"
#include "llvm/Support/Path.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace testing;

namespace {

class CollectingStreamer : public remark::detail::MLIRRemarkStreamerBase {
public:
  CollectingStreamer(std::vector<remark::detail::Remark> &remarks)
      : remarks(remarks) {}

  void streamOptimizationRemark(const remark::detail::Remark &remark) override {
    remarks.push_back(remark);
  }

private:
  std::vector<remark::detail::Remark> &remarks;
};

std::optional<std::string> getArg(const remark::detail::Remark &remark,
                                  StringRef key) {
  for (const remark::detail::Remark::Arg &arg : remark.getArgs())
    if (arg.key == key)
      return arg.val;
  return std::nullopt;
}

class LLVMRemarkImportTest : public ::testing::Test {
protected:
  void SetUp() override {
    // The nested module is the symbol named after the LLVM function.
    module = parseSourceString<ModuleOp>(R"mlir(
      module @anchor {
        module @kernel {
        }
        %0 = builtin.unrealized_conversion_cast to i32
      }
    )mlir",
                                         &context);
    ASSERT_TRUE(module);
    anchor = module->getOperation();
    kernel = SymbolTable::lookupSymbolIn(anchor, "kernel");
    ASSERT_TRUE(kernel);
    nonSymbolTable = &module->getBody()->back();
  }

  void enableRemarks(const remark::RemarkCategories &cats) {
    ASSERT_TRUE(succeeded(remark::enableOptimizationRemarks(
        context, std::make_unique<CollectingStreamer>(remarks),
        std::make_unique<remark::RemarkEmittingPolicyAll>(), cats)));
  }

  void installHandler(llvm::LLVMContext &llvmContext) {
    llvmContext.setDiagnosticHandler(
        std::make_unique<remark::LLVMToMLIRDiagnosticHandler>(anchor),
        /*RespectFilters=*/true);
  }

  llvm::Function *createKernel(llvm::Module &llvmModule) {
    llvm::FunctionType *type = llvm::FunctionType::get(
        llvm::Type::getVoidTy(llvmModule.getContext()), /*isVarArg=*/false);
    return llvm::Function::Create(type, llvm::GlobalValue::ExternalLinkage,
                                  "kernel", llvmModule);
  }

  MLIRContext context;
  OwningOpRef<ModuleOp> module;
  Operation *anchor = nullptr;
  Operation *kernel = nullptr;
  Operation *nonSymbolTable = nullptr;
  std::vector<remark::detail::Remark> remarks;
};

TEST_F(LLVMRemarkImportTest, HandlerAnswersRemarkQueriesFromEngine) {
  enableRemarks({/*all=*/std::nullopt, /*passed=*/"llvm-loop-unroll",
                 /*missed=*/"llvm-inline", /*analysis=*/std::nullopt,
                 /*failed=*/std::nullopt});

  llvm::LLVMContext llvmContext;
  installHandler(llvmContext);
  const llvm::DiagnosticHandler *handler = llvmContext.getDiagHandlerPtr();
  ASSERT_TRUE(handler);

  EXPECT_TRUE(handler->isAnyRemarkEnabled());
  EXPECT_TRUE(handler->isPassedOptRemarkEnabled("loop-unroll"));
  EXPECT_FALSE(handler->isPassedOptRemarkEnabled("inline"));
  EXPECT_TRUE(handler->isMissedOptRemarkEnabled("inline"));
  EXPECT_FALSE(handler->isMissedOptRemarkEnabled("loop-unroll"));
  EXPECT_FALSE(handler->isAnalysisRemarkEnabled("loop-unroll"));
  EXPECT_TRUE(handler->isAnyRemarkEnabled("loop-unroll"));
  EXPECT_FALSE(handler->isAnyRemarkEnabled("licm"));
}

TEST_F(LLVMRemarkImportTest, HandlerWithoutEngineDisablesRemarks) {
  llvm::LLVMContext llvmContext;
  installHandler(llvmContext);
  const llvm::DiagnosticHandler *handler = llvmContext.getDiagHandlerPtr();
  ASSERT_TRUE(handler);
  EXPECT_FALSE(handler->isAnyRemarkEnabled());
  EXPECT_FALSE(handler->isPassedOptRemarkEnabled("loop-unroll"));
  EXPECT_FALSE(handler->isMissedOptRemarkEnabled("loop-unroll"));
  EXPECT_FALSE(handler->isAnalysisRemarkEnabled("loop-unroll"));
}

TEST_F(LLVMRemarkImportTest, ImportsLiveRemarkWithFunctionLocation) {
  enableRemarks({/*all=*/std::nullopt, /*passed=*/"llvm-loop-unroll",
                 /*missed=*/std::nullopt, /*analysis=*/std::nullopt,
                 /*failed=*/std::nullopt});

  llvm::LLVMContext llvmContext;
  installHandler(llvmContext);
  llvm::Module llvmModule("m", llvmContext);
  llvm::Function *fn = createKernel(llvmModule);

  llvmContext.diagnose(
      llvm::OptimizationRemark("loop-unroll", "FullyUnrolled", fn)
      << "unrolled loop by a factor of "
      << llvm::DiagnosticInfoOptimizationBase::Argument("UnrollCount", 4));
  // Not enabled: missed remarks and the `inline` category.
  llvmContext.diagnose(
      llvm::OptimizationRemarkMissed("loop-unroll", "NotUnrolled", fn)
      << "not unrolled");
  llvmContext.diagnose(llvm::OptimizationRemark("inline", "Inlined", fn)
                       << "inlined");

  ASSERT_EQ(remarks.size(), 1u);
  const remark::detail::Remark &remark = remarks[0];
  EXPECT_EQ(remark.getRemarkKind(), remark::RemarkKind::RemarkPassed);
  EXPECT_EQ(remark.getRemarkName(), "FullyUnrolled");
  EXPECT_EQ(remark.getCategoryName(), "llvm-loop-unroll");
  EXPECT_EQ(remark.getFunction(), "kernel");
  EXPECT_EQ(remark.getLocation(), kernel->getLoc());
  EXPECT_EQ(getArg(remark, "Remark"), "unrolled loop by a factor of 4");
  EXPECT_EQ(getArg(remark, "UnrollCount"), "4");
  EXPECT_FALSE(getArg(remark, "String"));
}

TEST_F(LLVMRemarkImportTest, ImportsLiveRemarkWithDebugLocation) {
  enableRemarks({/*all=*/std::nullopt, /*passed=*/std::nullopt,
                 /*missed=*/"llvm-licm", /*analysis=*/std::nullopt,
                 /*failed=*/std::nullopt});

  llvm::LLVMContext llvmContext;
  installHandler(llvmContext);
  llvm::Module llvmModule("m", llvmContext);
  llvm::Function *fn = createKernel(llvmModule);

  llvm::DIBuilder dib(llvmModule);
  llvm::DIFile *file = dib.createFile("kernel.mlir", "/src");
  llvm::DICompileUnit *cu = dib.createCompileUnit(
      llvm::dwarf::DW_LANG_C, file, "mlir", /*isOptimized=*/false, "", 0);
  llvm::DISubroutineType *fnType =
      dib.createSubroutineType(dib.getOrCreateTypeArray({}));
  llvm::DISubprogram *sp = dib.createFunction(
      cu, "kernel", "kernel", file, /*LineNo=*/1, fnType, /*ScopeLine=*/1,
      llvm::DINode::FlagZero, llvm::DISubprogram::SPFlagDefinition);
  fn->setSubprogram(sp);
  llvm::BasicBlock *bb = llvm::BasicBlock::Create(llvmContext, "entry", fn);
  llvm::IRBuilder<> builder(bb);
  llvm::Instruction *inst = builder.CreateRetVoid();
  inst->setDebugLoc(llvm::DILocation::get(llvmContext, 7, 3, sp));
  dib.finalize();

  llvmContext.diagnose(
      llvm::OptimizationRemarkMissed("licm", "LoadNotHoisted", inst)
      << "failed to hoist load");

  ASSERT_EQ(remarks.size(), 1u);
  const remark::detail::Remark &remark = remarks[0];
  EXPECT_EQ(remark.getRemarkKind(), remark::RemarkKind::RemarkMissed);
  EXPECT_EQ(remark.getRemarkName(), "LoadNotHoisted");
  EXPECT_EQ(remark.getCategoryName(), "llvm-licm");
  auto loc = dyn_cast<FileLineColLoc>(remark.getLocation());
  ASSERT_TRUE(loc);
  SmallString<32> expectedFile;
  llvm::sys::path::append(expectedFile, "/src", "kernel.mlir");
  EXPECT_EQ(loc.getFilename().getValue(), expectedFile);
  EXPECT_EQ(loc.getLine(), 7u);
  EXPECT_EQ(loc.getColumn(), 3u);
}

TEST_F(LLVMRemarkImportTest, ImportsOptimizationFailureAsFailedRemark) {
  enableRemarks({/*all=*/std::nullopt, /*passed=*/std::nullopt,
                 /*missed=*/std::nullopt, /*analysis=*/std::nullopt,
                 /*failed=*/"llvm-transform-warning"});

  llvm::LLVMContext llvmContext;
  installHandler(llvmContext);
  llvm::Module llvmModule("m", llvmContext);
  llvm::Function *fn = createKernel(llvmModule);
  llvm::BasicBlock *bb = llvm::BasicBlock::Create(llvmContext, "entry", fn);

  llvmContext.diagnose(llvm::DiagnosticInfoOptimizationFailure(
                           "transform-warning", "FailedRequestedUnrolling",
                           llvm::DiagnosticLocation(), bb)
                       << "loop not unrolled");

  ASSERT_EQ(remarks.size(), 1u);
  EXPECT_EQ(remarks[0].getRemarkKind(), remark::RemarkKind::RemarkFailure);
  EXPECT_EQ(remarks[0].getRemarkName(), "FailedRequestedUnrolling");
  EXPECT_EQ(remarks[0].getCategoryName(), "llvm-transform-warning");
  EXPECT_EQ(getArg(remarks[0], "Remark"), "loop not unrolled");
}

TEST_F(LLVMRemarkImportTest, LeavesRemarksNotEnabledInEngineToLLVM) {
  enableRemarks({/*all=*/std::nullopt, /*passed=*/"llvm-loop-unroll",
                 /*missed=*/std::nullopt, /*analysis=*/std::nullopt,
                 /*failed=*/std::nullopt});

  llvm::LLVMContext llvmContext;
  llvm::Module llvmModule("m", llvmContext);
  llvm::Function *fn = createKernel(llvmModule);
  llvm::BasicBlock *bb = llvm::BasicBlock::Create(llvmContext, "entry", fn);
  remark::LLVMToMLIRDiagnosticHandler handler(anchor);

  EXPECT_TRUE(handler.handleDiagnostics(
      llvm::OptimizationRemark("loop-unroll", "FullyUnrolled", fn)
      << "unrolled"));
  EXPECT_FALSE(handler.handleDiagnostics(
      llvm::OptimizationRemarkMissed("loop-unroll", "NotUnrolled", fn)
      << "not unrolled"));
  EXPECT_FALSE(handler.handleDiagnostics(
      llvm::DiagnosticInfoOptimizationFailure("transform-warning",
                                              "FailedRequestedUnrolling",
                                              llvm::DiagnosticLocation(), bb)
      << "loop not unrolled"));
  EXPECT_EQ(remarks.size(), 1u);
}

TEST_F(LLVMRemarkImportTest, ReportsLLVMDiagnosticsAsMLIRDiagnostics) {
  std::vector<std::pair<DiagnosticSeverity, std::string>> diagnostics;
  std::vector<Location> locations;
  ScopedDiagnosticHandler diagHandler(&context, [&](Diagnostic &diag) {
    diagnostics.emplace_back(diag.getSeverity(), diag.str());
    locations.push_back(diag.getLocation());
    return success();
  });

  llvm::LLVMContext llvmContext;
  installHandler(llvmContext);
  llvm::Module llvmModule("m", llvmContext);
  llvm::Function *fn = createKernel(llvmModule);

  llvmContext.diagnose(
      llvm::DiagnosticInfoUnsupported(*fn, "unsupported operation"));
  llvmContext.diagnose(llvm::DiagnosticInfoUnsupported(
      *fn, "suspicious operation", llvm::DiagnosticLocation(),
      llvm::DS_Warning));

  ASSERT_EQ(diagnostics.size(), 2u);
  EXPECT_EQ(diagnostics[0].first, DiagnosticSeverity::Error);
  EXPECT_THAT(diagnostics[0].second, HasSubstr("unsupported operation"));
  EXPECT_THAT(diagnostics[0].second, Not(HasSubstr("<unknown>:0:0")));
  EXPECT_EQ(locations[0], kernel->getLoc());
  EXPECT_EQ(diagnostics[1].first, DiagnosticSeverity::Warning);
  EXPECT_THAT(diagnostics[1].second, HasSubstr("suspicious operation"));
  EXPECT_TRUE(llvmContext.getDiagHandlerPtr()->HasErrors);
}

TEST_F(LLVMRemarkImportTest, ImportsSerializedRemarks) {
  enableRemarks({/*all=*/"llvm-inline|llvm-loop-unroll", /*passed=*/"",
                 /*missed=*/"",
                 /*analysis=*/"", /*failed=*/""});

  const char *yaml = R"yaml(--- !Passed
Pass:            inline
Name:            Inlined
DebugLoc:        { File: kernel.mlir, Line: 12, Column: 5 }
Function:        kernel
Hotness:         300
Args:
  - String:          'inlined '
  - Callee:          helper
  - String:          ' into '
  - Caller:          kernel
  - Remark:          collides
...
--- !Missed
Pass:            loop-unroll
Name:            NotUnrolled
Function:        kernel
Args:
  - String:          'not unrolled: '
  - Reason:          trip count unknown
...
--- !Missed
Pass:            loop-unroll
Name:            NotUnrolled
Function:        not_a_symbol
...
--- !Analysis
Pass:            asm-printer
Name:            InstructionCount
Function:        kernel
Args:
  - NumInstructions: '42'
...
)yaml";

  ASSERT_TRUE(succeeded(
      remark::importLLVMRemarks(anchor, yaml, llvm::remarks::Format::YAML)));

  // The `asm-printer` remark is not enabled.
  ASSERT_EQ(remarks.size(), 3u);

  const remark::detail::Remark &passed = remarks[0];
  EXPECT_EQ(passed.getRemarkKind(), remark::RemarkKind::RemarkPassed);
  EXPECT_EQ(passed.getRemarkName(), "Inlined");
  EXPECT_EQ(passed.getCategoryName(), "llvm-inline");
  EXPECT_EQ(passed.getFunction(), "kernel");
  auto loc = dyn_cast<FileLineColLoc>(passed.getLocation());
  ASSERT_TRUE(loc);
  EXPECT_EQ(loc.getFilename().getValue(), "kernel.mlir");
  EXPECT_EQ(loc.getLine(), 12u);
  EXPECT_EQ(loc.getColumn(), 5u);
  EXPECT_EQ(getArg(passed, "Remark"), "inlined helper into kernelcollides");
  EXPECT_EQ(getArg(passed, "Callee"), "helper");
  EXPECT_EQ(getArg(passed, "Caller"), "kernel");
  EXPECT_EQ(getArg(passed, "LLVMRemark"), "collides");
  EXPECT_EQ(getArg(passed, "Hotness"), "300");
  EXPECT_FALSE(getArg(passed, "String"));

  const remark::detail::Remark &missed = remarks[1];
  EXPECT_EQ(missed.getRemarkKind(), remark::RemarkKind::RemarkMissed);
  EXPECT_EQ(missed.getRemarkName(), "NotUnrolled");
  EXPECT_EQ(missed.getCategoryName(), "llvm-loop-unroll");
  EXPECT_EQ(missed.getLocation(), kernel->getLoc());
  EXPECT_EQ(getArg(missed, "Remark"), "not unrolled: trip count unknown");
  EXPECT_EQ(getArg(missed, "Reason"), "trip count unknown");

  // Unknown function: attached to the anchor.
  EXPECT_EQ(remarks[2].getLocation(), anchor->getLoc());
}

TEST_F(LLVMRemarkImportTest, ImportsSerializedRemarksWithoutSymbolTable) {
  enableRemarks({/*all=*/"llvm-inline", /*passed=*/"", /*missed=*/"",
                 /*analysis=*/"", /*failed=*/""});
  const char *yaml = R"yaml(--- !Passed
Pass:            inline
Name:            Inlined
Function:        kernel
...
)yaml";
  ASSERT_TRUE(succeeded(remark::importLLVMRemarks(
      nonSymbolTable, yaml, llvm::remarks::Format::YAML)));
  ASSERT_EQ(remarks.size(), 1u);
  EXPECT_EQ(remarks[0].getLocation(), nonSymbolTable->getLoc());
}

TEST_F(LLVMRemarkImportTest, ImportSerializedRemarksReportsParseErrors) {
  enableRemarks({/*all=*/".*", /*passed=*/"", /*missed=*/"", /*analysis=*/"",
                 /*failed=*/""});
  EXPECT_TRUE(failed(remark::importLLVMRemarks(anchor, "--- !Passed\nFoo: [",
                                               llvm::remarks::Format::YAML)));
  EXPECT_TRUE(failed(remark::importLLVMRemarks(
      anchor, "not a bitstream", llvm::remarks::Format::Bitstream)));
  EXPECT_TRUE(remarks.empty());
}

TEST_F(LLVMRemarkImportTest, ImportSerializedRemarksWithoutEngine) {
  EXPECT_TRUE(succeeded(remark::importLLVMRemarks(
      anchor, "not a remark file", llvm::remarks::Format::YAML)));
}

} // namespace
