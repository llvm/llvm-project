//===- llvm/unittest/Bitcode/BitReaderTest.cpp - Tests for BitReader ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/Bitcode/BitstreamReader.h"
#include "llvm/Bitcode/BitstreamWriter.h"
#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/DataStream.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/StreamingMemoryObject.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

std::unique_ptr<Module> parseAssembly(const char *Assembly) {
  SMDiagnostic Error;
  std::unique_ptr<Module> M =
      parseAssemblyString(Assembly, Error, getGlobalContext());

  std::string ErrMsg;
  raw_string_ostream OS(ErrMsg);
  Error.print("", OS);

  // A failure here means that the test itself is buggy.
  if (!M)
    report_fatal_error(OS.str().c_str());

  return M;
}

static void writeModuleToBuffer(std::unique_ptr<Module> Mod,
                                SmallVectorImpl<char> &Buffer) {
  raw_svector_ostream OS(Buffer);
  WriteBitcodeToFile(Mod.get(), OS);
}

static std::unique_ptr<Module> getLazyModuleFromAssembly(LLVMContext &Context,
                                                         SmallString<1024> &Mem,
                                                         const char *Assembly) {
  writeModuleToBuffer(parseAssembly(Assembly), Mem);
  std::unique_ptr<MemoryBuffer> Buffer =
      MemoryBuffer::getMemBuffer(Mem.str(), "test", false);
  ErrorOr<std::unique_ptr<Module>> ModuleOrErr =
      getLazyBitcodeModule(std::move(Buffer), Context);
  return std::move(ModuleOrErr.get());
}

class BufferDataStreamer : public DataStreamer {
  std::unique_ptr<MemoryBuffer> Buffer;
  unsigned Pos = 0;
  size_t GetBytes(unsigned char *Out, size_t Len) override {
    StringRef Buf = Buffer->getBuffer();
    size_t Left = Buf.size() - Pos;
    Len = std::min(Left, Len);
    memcpy(Out, Buffer->getBuffer().substr(Pos).data(), Len);
    Pos += Len;
    return Len;
  }

public:
  BufferDataStreamer(std::unique_ptr<MemoryBuffer> Buffer)
      : Buffer(std::move(Buffer)) {}
};

static std::unique_ptr<Module>
getStreamedModuleFromAssembly(LLVMContext &Context, SmallString<1024> &Mem,
                              const char *Assembly) {
  writeModuleToBuffer(parseAssembly(Assembly), Mem);
  std::unique_ptr<MemoryBuffer> Buffer =
      MemoryBuffer::getMemBuffer(Mem.str(), "test", false);
  auto Streamer = llvm::make_unique<BufferDataStreamer>(std::move(Buffer));
  ErrorOr<std::unique_ptr<Module>> ModuleOrErr =
      getStreamedBitcodeModule("test", std::move(Streamer), Context);
  return std::move(ModuleOrErr.get());
}

// Checks if we correctly detect eof if we try to read N bits when there are not
// enough bits left on the input stream to read N bits, and we are using a data
// streamer. In particular, it checks if we properly set the object size when
// the eof is reached under such conditions.
TEST(BitReaderTest, TestForEofAfterReadFailureOnDataStreamer) {
  // Note: Because StreamingMemoryObject does a call to method GetBytes in it's
  // constructor, using internal constant kChunkSize, we must fill the input
  // with more characters than that amount.
  static size_t InputSize = StreamingMemoryObject::kChunkSize + 5;
  char *Text = new char[InputSize];
  std::memset(Text, 'a', InputSize);
  Text[InputSize - 1] = '\0';
  StringRef Input(Text);

  // Build bitsteam reader using data streamer.
  auto MemoryBuf = MemoryBuffer::getMemBuffer(Input);
  std::unique_ptr<DataStreamer> Streamer(
      new BufferDataStreamer(std::move(MemoryBuf)));
  auto OwnedBytes =
      llvm::make_unique<StreamingMemoryObject>(std::move(Streamer));
  auto Reader = llvm::make_unique<BitstreamReader>(std::move(OwnedBytes));
  BitstreamCursor Cursor;
  Cursor.init(Reader.get());

  // Jump to two bytes before end of stream.
  Cursor.JumpToBit((InputSize - 4) * CHAR_BIT);
  // Try to read 4 bytes when only 2 are present, resulting in error value 0.
  const size_t ReadErrorValue = 0;
  EXPECT_EQ(ReadErrorValue, Cursor.Read(32));
  // Should be at eof now.
  EXPECT_TRUE(Cursor.AtEndOfStream());

  delete[] Text;
}

TEST(BitReaderTest, MateralizeForwardRefWithStream) {
  SmallString<1024> Mem;

  LLVMContext Context;
  std::unique_ptr<Module> M = getStreamedModuleFromAssembly(
      Context, Mem, "@table = constant i8* blockaddress(@func, %bb)\n"
                    "define void @func() {\n"
                    "  unreachable\n"
                    "bb:\n"
                    "  unreachable\n"
                    "}\n");
  EXPECT_FALSE(M->getFunction("func")->empty());
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
  H->materialize();
  EXPECT_TRUE(F->empty());
  EXPECT_TRUE(G->empty());
  EXPECT_FALSE(H->empty());
  EXPECT_TRUE(J->empty());
  EXPECT_FALSE(verifyModule(*M, &dbgs()));

  // Materialize g.
  G->materialize();
  EXPECT_TRUE(F->empty());
  EXPECT_FALSE(G->empty());
  EXPECT_FALSE(H->empty());
  EXPECT_TRUE(J->empty());
  EXPECT_FALSE(verifyModule(*M, &dbgs()));

  // Materialize j.
  J->materialize();
  EXPECT_TRUE(F->empty());
  EXPECT_FALSE(G->empty());
  EXPECT_FALSE(H->empty());
  EXPECT_FALSE(J->empty());
  EXPECT_FALSE(verifyModule(*M, &dbgs()));

  // Materialize f.
  F->materialize();
  EXPECT_FALSE(F->empty());
  EXPECT_FALSE(G->empty());
  EXPECT_FALSE(H->empty());
  EXPECT_FALSE(J->empty());
  EXPECT_FALSE(verifyModule(*M, &dbgs()));
}

TEST(BitReaderTest, MaterializeFunctionsForBlockAddr) { // PR11677
  SmallString<1024> Mem;

  LLVMContext Context;
  std::unique_ptr<Module> M = getLazyModuleFromAssembly(
      Context, Mem, "@table = constant i8* blockaddress(@func, %bb)\n"
                    "define void @func() {\n"
                    "  unreachable\n"
                    "bb:\n"
                    "  unreachable\n"
                    "}\n");
  EXPECT_FALSE(verifyModule(*M, &dbgs()));
}

TEST(BitReaderTest, MaterializeFunctionsForBlockAddrInFunctionBefore) {
  SmallString<1024> Mem;

  LLVMContext Context;
  std::unique_ptr<Module> M = getLazyModuleFromAssembly(
      Context, Mem, "define i8* @before() {\n"
                    "  ret i8* blockaddress(@func, %bb)\n"
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
                    "define i8* @after() {\n"
                    "  ret i8* blockaddress(@func, %bb)\n"
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

} // end namespace
