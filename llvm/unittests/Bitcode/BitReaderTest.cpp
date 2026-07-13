//===- llvm/unittest/Bitcode/BitReaderTest.cpp - Tests for BitReader ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "BitReaderTestCode.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DebugProgramInstruction.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

std::unique_ptr<Module> parseAssembly(LLVMContext &Context,
                                      const char *Assembly) {
  SMDiagnostic Error;
  std::unique_ptr<Module> M = parseAssemblyString(Assembly, Error, Context);

  std::string ErrMsg;
  raw_string_ostream OS(ErrMsg);
  Error.print("", OS);

  // A failure here means that the test itself is buggy.
  if (!M)
    report_fatal_error(ErrMsg.c_str());

  return M;
}

static void writeModuleToBuffer(std::unique_ptr<Module> Mod,
                                SmallVectorImpl<char> &Buffer) {
  raw_svector_ostream OS(Buffer);
  WriteBitcodeToFile(*Mod, OS);
}

static std::unique_ptr<Module> getLazyModuleFromAssembly(LLVMContext &Context,
                                                         SmallString<1024> &Mem,
                                                         const char *Assembly) {
  writeModuleToBuffer(parseAssembly(Context, Assembly), Mem);
  Expected<std::unique_ptr<Module>> ModuleOrErr =
      getLazyBitcodeModule(MemoryBufferRef(Mem.str(), "test"), Context);
  if (!ModuleOrErr)
    report_fatal_error("Could not parse bitcode module");
  return std::move(ModuleOrErr.get());
}

// Tests that lazy evaluation can parse functions out of order.
TEST(BitReaderTest, MaterializeFunctionsOutOfOrder) {
  SmallString<1024> Mem;
  LLVMContext Context;
  std::unique_ptr<Module> M = getLazyModuleFromAssembly(
      Context, Mem, "define void @f() {\n"
                    "  unreachable\n"
                    "}\n"
                    "define void @g() {\n"
                    "  unreachable\n"
                    "}\n"
                    "define void @h() {\n"
                    "  unreachable\n"
                    "}\n"
                    "define void @j() {\n"
                    "  unreachable\n"
                    "}\n");
  EXPECT_FALSE(verifyModule(*M, &dbgs()));

  Function *F = M->getFunction("f");
  Function *G = M->getFunction("g");
  Function *H = M->getFunction("h");
  Function *J = M->getFunction("j");

  // Initially all functions are not materialized (no basic blocks).
  EXPECT_TRUE(F->empty());
  EXPECT_TRUE(G->empty());
  EXPECT_TRUE(H->empty());
  EXPECT_TRUE(J->empty());
  EXPECT_FALSE(verifyModule(*M, &dbgs()));

  // Materialize h.
  ASSERT_FALSE(H->materialize());
  EXPECT_TRUE(F->empty());
  EXPECT_TRUE(G->empty());
  EXPECT_FALSE(H->empty());
  EXPECT_TRUE(J->empty());
  EXPECT_FALSE(verifyModule(*M, &dbgs()));

  // Materialize g.
  ASSERT_FALSE(G->materialize());
  EXPECT_TRUE(F->empty());
  EXPECT_FALSE(G->empty());
  EXPECT_FALSE(H->empty());
  EXPECT_TRUE(J->empty());
  EXPECT_FALSE(verifyModule(*M, &dbgs()));

  // Materialize j.
  ASSERT_FALSE(J->materialize());
  EXPECT_TRUE(F->empty());
  EXPECT_FALSE(G->empty());
  EXPECT_FALSE(H->empty());
  EXPECT_FALSE(J->empty());
  EXPECT_FALSE(verifyModule(*M, &dbgs()));

  // Materialize f.
  ASSERT_FALSE(F->materialize());
  EXPECT_FALSE(F->empty());
  EXPECT_FALSE(G->empty());
  EXPECT_FALSE(H->empty());
  EXPECT_FALSE(J->empty());
  EXPECT_FALSE(verifyModule(*M, &dbgs()));
}

TEST(BitReaderTest, MaterializeFunctionsStrictFP) {
  SmallString<1024> Mem;

  LLVMContext Context;
  std::unique_ptr<Module> M = getLazyModuleFromAssembly(
      Context, Mem, "define double @foo(double %a) {\n"
                    "  %result = call double @bar(double %a) strictfp\n"
                    "  ret double %result\n"
                    "}\n"
                    "declare double @bar(double)\n");
  Function *Foo = M->getFunction("foo");
  ASSERT_FALSE(Foo->materialize());
  EXPECT_FALSE(Foo->empty());

  for (auto &BB : *Foo) {
    auto It = BB.begin();
    while (It != BB.end()) {
      Instruction &I = *It;
      ++It;

      if (auto *Call = dyn_cast<CallBase>(&I)) {
        EXPECT_FALSE(Call->isStrictFP());
        EXPECT_TRUE(Call->isNoBuiltin());
      }
    }
  }

  EXPECT_FALSE(verifyModule(*M, &dbgs()));
}

TEST(BitReaderTest, MaterializeConstrainedFPStrictFP) {
  SmallString<1024> Mem;

  LLVMContext Context;
  std::unique_ptr<Module> M = getLazyModuleFromAssembly(
      Context, Mem,
      "define double @foo(double %a) strictfp {\n"
      "  %result = call double @llvm.experimental.constrained.sqrt.f64(double "
      "%a, metadata !\"round.tonearest\", metadata !\"fpexcept.strict\") "
      "strictfp\n"
      "  ret double %result\n"
      "}\n"
      "declare double @llvm.experimental.constrained.sqrt.f64(double, "
      "metadata, metadata)\n");
  Function *Foo = M->getFunction("foo");
  ASSERT_FALSE(Foo->materialize());
  EXPECT_FALSE(Foo->empty());

  for (auto &BB : *Foo) {
    auto It = BB.begin();
    while (It != BB.end()) {
      Instruction &I = *It;
      ++It;

      if (auto *Call = dyn_cast<CallBase>(&I)) {
        EXPECT_TRUE(Call->isStrictFP());
        EXPECT_FALSE(Call->isNoBuiltin());
      }
    }
  }

  EXPECT_FALSE(verifyModule(*M, &dbgs()));
}

TEST(BitReaderTest, MaterializeFunctionsForBlockAddr) { // PR11677
  SmallString<1024> Mem;

  LLVMContext Context;
  std::unique_ptr<Module> M = getLazyModuleFromAssembly(
      Context, Mem, "@table = constant ptr blockaddress(@func, %bb)\n"
                    "define void @func() {\n"
                    "  unreachable\n"
                    "bb:\n"
                    "  unreachable\n"
                    "}\n");
  EXPECT_FALSE(verifyModule(*M, &dbgs()));
  EXPECT_FALSE(M->getFunction("func")->empty());
}

TEST(BitReaderTest, MaterializeFunctionsForBlockAddrInFunctionBefore) {
  SmallString<1024> Mem;

  LLVMContext Context;
  std::unique_ptr<Module> M = getLazyModuleFromAssembly(
      Context, Mem, "define ptr @before() {\n"
                    "  ret ptr blockaddress(@func, %bb)\n"
                    "}\n"
                    "define void @other() {\n"
                    "  unreachable\n"
                    "}\n"
                    "define void @func() {\n"
                    "  unreachable\n"
                    "bb:\n"
                    "  unreachable\n"
                    "}\n");
  EXPECT_TRUE(M->getFunction("before")->empty());
  EXPECT_TRUE(M->getFunction("func")->empty());
  EXPECT_FALSE(verifyModule(*M, &dbgs()));

  // Materialize @before, pulling in @func.
  EXPECT_FALSE(M->getFunction("before")->materialize());
  EXPECT_FALSE(M->getFunction("func")->empty());
  EXPECT_TRUE(M->getFunction("other")->empty());
  EXPECT_FALSE(verifyModule(*M, &dbgs()));
}

TEST(BitReaderTest, MaterializeFunctionsForBlockAddrInFunctionAfter) {
  SmallString<1024> Mem;

  LLVMContext Context;
  std::unique_ptr<Module> M = getLazyModuleFromAssembly(
      Context, Mem, "define void @func() {\n"
                    "  unreachable\n"
                    "bb:\n"
                    "  unreachable\n"
                    "}\n"
                    "define void @other() {\n"
                    "  unreachable\n"
                    "}\n"
                    "define ptr @after() {\n"
                    "  ret ptr blockaddress(@func, %bb)\n"
                    "}\n");
  EXPECT_TRUE(M->getFunction("after")->empty());
  EXPECT_TRUE(M->getFunction("func")->empty());
  EXPECT_FALSE(verifyModule(*M, &dbgs()));

  // Materialize @after, pulling in @func.
  EXPECT_FALSE(M->getFunction("after")->materialize());
  EXPECT_FALSE(M->getFunction("func")->empty());
  EXPECT_TRUE(M->getFunction("other")->empty());
  EXPECT_FALSE(verifyModule(*M, &dbgs()));
}

// Helper function to convert type metadata to a string for testing
static std::string mdToString(Metadata *MD) {
  std::string S;
  if (auto *VMD = dyn_cast<ValueAsMetadata>(MD)) {
    if (VMD->getType()->isPointerTy()) {
      S += "ptr";
      return S;
    }
  }

  if (auto *TMD = dyn_cast<MDTuple>(MD)) {
    S += "!{";
    for (unsigned I = 0; I < TMD->getNumOperands(); I++) {
      if (I != 0)
        S += ", ";
      S += mdToString(TMD->getOperand(I).get());
    }
    S += "}";
  } else if (auto *SMD = dyn_cast<MDString>(MD)) {
    S += "!'";
    S += SMD->getString();
    S += "'";
  } else if (auto *I = mdconst::dyn_extract<ConstantInt>(MD)) {
    S += std::to_string(I->getZExtValue());
  } else if (auto *P = mdconst::dyn_extract<PoisonValue>(MD)) {
    auto *Ty = P->getType();
    if (Ty->isIntegerTy()) {
      S += "i";
      S += std::to_string(Ty->getIntegerBitWidth());
    } else if (Ty->isStructTy()) {
      S += "%";
      S += Ty->getStructName();
    } else {
      llvm_unreachable("unhandled poison metadata");
    }
  } else {
    llvm_unreachable("unhandled metadata");
  }
  return S;
}

// Recursively look into a (pointer) type and the the type.
// For primitive types it's a poison value of the type, for a pointer it's a
// metadata tuple with the addrspace and the referenced type. For a function,
// it's a tuple where the first element is the string "function", the second
// element is the return type or the string "void" and the following elements
// are the argument types.
static Metadata *getTypeMetadataEntry(unsigned TypeID, LLVMContext &Context,
                                      GetTypeByIDTy GetTypeByID,
                                      GetContainedTypeIDTy GetContainedTypeID) {
  Type *Ty = GetTypeByID(TypeID);
  if (auto *FTy = dyn_cast<FunctionType>(Ty)) {
    // Save the function signature as metadata
    SmallVector<Metadata *> SignatureMD;
    SignatureMD.push_back(MDString::get(Context, "function"));
    // Return type
    if (FTy->getReturnType()->isVoidTy())
      SignatureMD.push_back(MDString::get(Context, "void"));
    else
      SignatureMD.push_back(getTypeMetadataEntry(GetContainedTypeID(TypeID, 0),
                                                 Context, GetTypeByID,
                                                 GetContainedTypeID));
    // Arguments
    for (unsigned I = 0; I != FTy->getNumParams(); ++I)
      SignatureMD.push_back(
          getTypeMetadataEntry(GetContainedTypeID(TypeID, I + 1), Context,
                               GetTypeByID, GetContainedTypeID));

    return MDTuple::get(Context, SignatureMD);
  }

  if (!Ty->isPointerTy())
    return ConstantAsMetadata::get(PoisonValue::get(Ty));

  // Return !{<addrspace>, <inner>} for pointer
  SmallVector<Metadata *, 2> MD;
  MD.push_back(ConstantAsMetadata::get(ConstantInt::get(
      Type::getInt32Ty(Context), Ty->getPointerAddressSpace())));
  MD.push_back(getTypeMetadataEntry(GetContainedTypeID(TypeID, 0), Context,
                                    GetTypeByID, GetContainedTypeID));
  return MDTuple::get(Context, MD);
}

// Test that when reading bitcode with typed pointers and upgrading them to
// opaque pointers, the type information of function signatures can be extracted
// and stored in metadata.
TEST(BitReaderTest, AccessFunctionTypeInfo) {
  StringRef Bitcode(reinterpret_cast<const char *>(AccessFunctionTypeInfoBc),
                    sizeof(AccessFunctionTypeInfoBc));

  LLVMContext Context;
  ParserCallbacks Callbacks;
  // Supply a callback that stores the signature of a function into metadata,
  // so that the types behind pointers can be accessed.
  // Each function gets a !types metadata, which is a tuple with one element
  // for a non-void return type and every argument. For primitive types it's
  // a poison value of the type, for a pointer it's a metadata tuple with
  // the addrspace and the referenced type.
  Callbacks.ValueType = [&](Value *V, unsigned TypeID,
                            GetTypeByIDTy GetTypeByID,
                            GetContainedTypeIDTy GetContainedTypeID) {
    if (auto *F = dyn_cast<Function>(V)) {
      auto *MD = getTypeMetadataEntry(TypeID, F->getContext(), GetTypeByID,
                                      GetContainedTypeID);
      F->setMetadata("types", cast<MDNode>(MD));
    }
  };

  Expected<std::unique_ptr<Module>> ModuleOrErr =
      parseBitcodeFile(MemoryBufferRef(Bitcode, "test"), Context, Callbacks);

  if (!ModuleOrErr)
    report_fatal_error("Could not parse bitcode module");
  std::unique_ptr<Module> M = std::move(ModuleOrErr.get());

  EXPECT_EQ(mdToString(M->getFunction("func")->getMetadata("types")),
            "!{!'function', !'void'}");
  EXPECT_EQ(mdToString(M->getFunction("func_header")->getMetadata("types")),
            "!{!'function', i32}");
  EXPECT_EQ(mdToString(M->getFunction("ret_ptr")->getMetadata("types")),
            "!{!'function', !{0, i8}}");
  EXPECT_EQ(mdToString(M->getFunction("ret_and_arg_ptr")->getMetadata("types")),
            "!{!'function', !{0, i8}, !{8, i32}}");
  EXPECT_EQ(mdToString(M->getFunction("double_ptr")->getMetadata("types")),
            "!{!'function', !{1, i8}, !{2, !{0, i32}}, !{0, !{0, !{0, i32}}}}");
}

// Test that when reading bitcode with typed pointers and upgrading them to
// opaque pointers, the type information of pointers in metadata can be
// extracted and stored in metadata.
TEST(BitReaderTest, AccessMetadataTypeInfo) {
  StringRef Bitcode(reinterpret_cast<const char *>(AccessMetadataTypeInfoBc),
                    sizeof(AccessFunctionTypeInfoBc));

  LLVMContext Context;
  ParserCallbacks Callbacks;
  // Supply a callback that stores types from metadata,
  // so that the types behind pointers can be accessed.
  // Non-pointer entries are ignored. Values with a pointer type are
  // replaced by a metadata tuple with {original value, type md}. We cannot
  // save the metadata outside because after conversion to opaque pointers,
  // entries are not distinguishable anymore (e.g. i32* and i8* are both
  // upgraded to ptr).
  Callbacks.MDType = [&](Metadata **Val, unsigned TypeID,
                         GetTypeByIDTy GetTypeByID,
                         GetContainedTypeIDTy GetContainedTypeID) {
    auto *OrigVal = cast<ValueAsMetadata>(*Val);
    if (OrigVal->getType()->isPointerTy()) {
      // Ignore function references, their signature can be saved like
      // in the test above
      if (!isa<Function>(OrigVal->getValue())) {
        SmallVector<Metadata *> Tuple;
        Tuple.push_back(OrigVal);
        Tuple.push_back(getTypeMetadataEntry(GetContainedTypeID(TypeID, 0),
                                             OrigVal->getContext(), GetTypeByID,
                                             GetContainedTypeID));
        *Val = MDTuple::get(OrigVal->getContext(), Tuple);
      }
    }
  };

  Expected<std::unique_ptr<Module>> ModuleOrErr =
      parseBitcodeFile(MemoryBufferRef(Bitcode, "test"), Context, Callbacks);

  if (!ModuleOrErr)
    report_fatal_error("Could not parse bitcode module");
  std::unique_ptr<Module> M = std::move(ModuleOrErr.get());

  EXPECT_EQ(
      mdToString(M->getNamedMetadata("md")->getOperand(0)),
      "!{2, !{ptr, %dx.types.f32}, ptr, !{ptr, !{!'function', !'void'}}}");
  EXPECT_EQ(mdToString(M->getNamedMetadata("md2")->getOperand(0)),
            "!{!{ptr, !{!'function', !{0, i8}, !{2, !{0, i32}}}}, !{ptr, !{0, "
            "!{0, i32}}}}");
}

// Counts calls to the given debug intrinsic in F.
static unsigned countIntrinsicCalls(Function &F, Intrinsic::ID ID) {
  unsigned Count = 0;
  for (Instruction &I : instructions(F))
    if (auto *II = dyn_cast<IntrinsicInst>(&I))
      if (II->getIntrinsicID() == ID)
        Count++;
  return Count;
}

// Per-kind tally of the non-instruction debug records.
struct DebugRecordCounts {
  unsigned Values = 0;
  unsigned Declares = 0;
  unsigned Assigns = 0;
  unsigned Labels = 0;
  unsigned total() const { return Values + Declares + Assigns + Labels; }
};

static DebugRecordCounts countDebugRecords(Function &F) {
  DebugRecordCounts Counts;
  for (Instruction &I : instructions(F)) {
    for (DbgRecord &DR : I.getDbgRecordRange()) {
      if (auto *DVR = dyn_cast<DbgVariableRecord>(&DR)) {
        if (DVR->isDbgAssign())
          Counts.Assigns++;
        else if (DVR->isDbgDeclare())
          Counts.Declares++;
        else if (DVR->isDbgValue())
          Counts.Values++;
      } else if (isa<DbgLabelRecord>(&DR)) {
        Counts.Labels++;
      }
    }
  }
  return Counts;
}

// The custom metadata kind name and contents attached to the dbg.* calls
// before serialization. The contents are checked back for integrity.
static constexpr StringRef CustomMDKind = "custom.test";
static constexpr StringRef CustomMDString = "preserved-by-skip";
static constexpr uint64_t CustomMDInt = 42;

static bool isLegacyDbgIntrinsic(const Instruction &I) {
  const IntrinsicInst *II = dyn_cast<IntrinsicInst>(&I);
  if (!II)
    return false;
  switch (II->getIntrinsicID()) {
  case Intrinsic::dbg_value:
  case Intrinsic::dbg_declare:
  case Intrinsic::dbg_assign:
  case Intrinsic::dbg_label:
    return true;
  }
  return false;
}

// Attaches CustomMD (an MDString + an integer) to every dbg.* call in F.
static void attachCustomMetadataToDbgInsts(Function &F) {
  LLVMContext &Ctx = F.getContext();
  MDNode *Custom = MDNode::get(Ctx, {MDString::get(Ctx, CustomMDString),
                                     ConstantAsMetadata::get(ConstantInt::get(
                                         Type::getInt32Ty(Ctx), CustomMDInt))});
  for (Instruction &I : instructions(F))
    if (isLegacyDbgIntrinsic(I))
      I.setMetadata(CustomMDKind, Custom);
}

// Verifies the custom metadata survived round-tripping and still holds the
// exact MDString + integer it was created with.
static void checkDbgInstCustomMetadata(Function &F) {
  for (Instruction &I : instructions(F)) {
    if (!isLegacyDbgIntrinsic(I))
      continue;

    MDNode *MD = I.getMetadata(CustomMDKind);
    ASSERT_NE(MD, nullptr);
    ASSERT_EQ(MD->getNumOperands(), 2u);

    auto *Str = dyn_cast<MDString>(MD->getOperand(0));
    ASSERT_NE(Str, nullptr);
    EXPECT_EQ(Str->getString(), CustomMDString);

    auto *Int = mdconst::dyn_extract<ConstantInt>(MD->getOperand(1));
    ASSERT_NE(Int, nullptr);
    EXPECT_EQ(Int->getZExtValue(), CustomMDInt);
  }
}

// IR exercising every debug intrinsic kind: dbg.declare, dbg.value, dbg.assign
// (linked to the alloca via DIAssignID) and dbg.label. parseAssembly leaves the
// module in the new debug records format; the test below lowers it back to
// intrinsic form before writing, so the resulting bitcode encodes the debug
// info as @llvm.dbg.* intrinsic calls.
static constexpr char DebugIntrinsicAssembly[] = R"(
  define i32 @f(i32 %p) !dbg !6 {
  entry:
    %a = alloca i32, align 4, !DIAssignID !14
    call void @llvm.dbg.declare(metadata ptr %a, metadata !9, metadata !DIExpression()), !dbg !13
    call void @llvm.dbg.value(metadata i32 %p, metadata !10, metadata !DIExpression()), !dbg !13
    call void @llvm.dbg.assign(metadata i32 %p, metadata !11, metadata !DIExpression(), metadata !14, metadata ptr %a, metadata !DIExpression()), !dbg !13
    call void @llvm.dbg.label(metadata !15), !dbg !13
    store i32 %p, ptr %a, align 4, !dbg !13
    ret i32 %p, !dbg !13
  }
  declare void @llvm.dbg.declare(metadata, metadata, metadata)
  declare void @llvm.dbg.value(metadata, metadata, metadata)
  declare void @llvm.dbg.assign(metadata, metadata, metadata, metadata, metadata, metadata)
  declare void @llvm.dbg.label(metadata)

  !llvm.dbg.cu = !{!0}
  !llvm.module.flags = !{!5}

  !0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "test", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
  !1 = !DIFile(filename: "t.ll", directory: "/")
  !2 = !{}
  !5 = !{i32 2, !"Debug Info Version", i32 3}
  !6 = distinct !DISubprogram(name: "f", linkageName: "f", scope: null, file: !1, line: 1, type: !7, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !8)
  !7 = !DISubroutineType(types: !2)
  !8 = !{!9, !10, !11}
  !9 = !DILocalVariable(name: "a", scope: !6, file: !1, line: 1, type: !12)
  !10 = !DILocalVariable(name: "p", scope: !6, file: !1, line: 1, type: !12)
  !11 = !DILocalVariable(name: "i", scope: !6, file: !1, line: 1, type: !12)
  !12 = !DIBasicType(name: "ty32", size: 32, encoding: DW_ATE_signed)
  !13 = !DILocation(line: 1, column: 1, scope: !6)
  !14 = distinct !DIAssignID()
  !15 = !DILabel(scope: !6, name: "label", file: !1, line: 1)
)";

// Asserts F holds exactly the intrinsic-form debug info (one call of each
// kind) and no debug records.
static void expectIntrinsicForm(Function &F) {
  EXPECT_EQ(countIntrinsicCalls(F, Intrinsic::dbg_declare), 1u);
  EXPECT_EQ(countIntrinsicCalls(F, Intrinsic::dbg_value), 1u);
  EXPECT_EQ(countIntrinsicCalls(F, Intrinsic::dbg_assign), 1u);
  EXPECT_EQ(countIntrinsicCalls(F, Intrinsic::dbg_label), 1u);
  EXPECT_EQ(countDebugRecords(F).total(), 0u);
}

// Asserts F holds exactly the record-form debug info (one record of each kind)
// and no debug intrinsic calls.
static void expectRecordForm(Function &F) {
  EXPECT_EQ(countIntrinsicCalls(F, Intrinsic::dbg_declare), 0u);
  EXPECT_EQ(countIntrinsicCalls(F, Intrinsic::dbg_value), 0u);
  EXPECT_EQ(countIntrinsicCalls(F, Intrinsic::dbg_assign), 0u);
  EXPECT_EQ(countIntrinsicCalls(F, Intrinsic::dbg_label), 0u);
  DebugRecordCounts Counts = countDebugRecords(F);
  EXPECT_EQ(Counts.Declares, 1u);
  EXPECT_EQ(Counts.Values, 1u);
  EXPECT_EQ(Counts.Assigns, 1u);
  EXPECT_EQ(Counts.Labels, 1u);
}

// ParserCallbacks::SkipDebugIntrinsicUpgrade controls whether the bitcode
// reader auto-upgrades debug intrinsic calls (llvm.dbg.*) to non-instruction
// debug records. Read the same intrinsic-form bitcode both ways and check that
// the flag toggles the upgrade for every debug intrinsic kind, and that custom
// metadata attached to a preserved intrinsic survives intact.
TEST(BitReaderTest, SkipDebugIntrinsicUpgrade) {
  SmallString<1024> Mem;
  {
    LLVMContext Context;
    std::unique_ptr<Module> M = parseAssembly(Context, DebugIntrinsicAssembly);
    // Lower the debug records produced by the parser back to intrinsic calls so
    // the bitcode we write out encodes llvm.dbg.* intrinsics.
    M->convertFromNewDbgValues();
    Function *F = M->getFunction("f");
    ASSERT_NE(F, nullptr);
    expectIntrinsicForm(*F);
    // Attach a custom metadata node to the dbg.value call. Records carry no
    // arbitrary attachments, so this only survives if the intrinsic does.
    attachCustomMetadataToDbgInsts(*F);
    writeModuleToBuffer(std::move(M), Mem);
  }

  // Default behavior: every debug intrinsic is upgraded to a debug record.
  {
    LLVMContext Context;
    Expected<std::unique_ptr<Module>> ModuleOrErr =
        parseBitcodeFile(MemoryBufferRef(Mem.str(), "test"), Context);
    if (!ModuleOrErr)
      report_fatal_error("Could not parse bitcode module");
    std::unique_ptr<Module> M = std::move(ModuleOrErr.get());

    Function *F = M->getFunction("f");
    ASSERT_NE(F, nullptr);
    expectRecordForm(*F);

    bool BrokenDebugInfo = false;
    EXPECT_FALSE(verifyModule(*M, &dbgs(), &BrokenDebugInfo));
    EXPECT_FALSE(BrokenDebugInfo);
  }

  // When SkipDebugIntrinsicUpgrade is true, the intrinsic-form debug info is
  // preserved for every kind and the caller is left responsible for upgrading
  // it.
  {
    LLVMContext Context;
    ParserCallbacks Callbacks;
    Callbacks.SkipDebugIntrinsicUpgrade = true;
    Expected<std::unique_ptr<Module>> ModuleOrErr = parseBitcodeFile(
        MemoryBufferRef(Mem.str(), "test"), Context, Callbacks);
    if (!ModuleOrErr)
      report_fatal_error("Could not parse bitcode module");
    std::unique_ptr<Module> M = std::move(ModuleOrErr.get());

    Function *F = M->getFunction("f");
    ASSERT_NE(F, nullptr);
    expectIntrinsicForm(*F);
    checkDbgInstCustomMetadata(*F);

    bool BrokenDebugInfo = false;
    EXPECT_FALSE(verifyModule(*M, &dbgs(), &BrokenDebugInfo));
    EXPECT_FALSE(BrokenDebugInfo);

    // Round-trip: writing the preserved intrinsic-form module back out and
    // reading it again with the flag set must still yield intrinsic-form debug
    // info, with the custom metadata intact. The reader must not silently
    // change the format on the way through.
    SmallString<1024> RoundTripMem;
    writeModuleToBuffer(std::move(M), RoundTripMem);

    LLVMContext RoundTripContext;
    Expected<std::unique_ptr<Module>> RoundTripOrErr =
        parseBitcodeFile(MemoryBufferRef(RoundTripMem.str(), "test"),
                         RoundTripContext, Callbacks);
    if (!RoundTripOrErr)
      report_fatal_error("Could not parse bitcode module");
    std::unique_ptr<Module> RoundTripM = std::move(RoundTripOrErr.get());

    Function *RoundTripF = RoundTripM->getFunction("f");
    ASSERT_NE(RoundTripF, nullptr);
    expectIntrinsicForm(*RoundTripF);
    checkDbgInstCustomMetadata(*RoundTripF);

    EXPECT_FALSE(verifyModule(*RoundTripM, &dbgs(), &BrokenDebugInfo));
    EXPECT_FALSE(BrokenDebugInfo);

    // The caller can still upgrade manually, matching the default behavior.
    RoundTripM->convertToNewDbgValues();
    expectRecordForm(*RoundTripF);
  }
}

} // end namespace
