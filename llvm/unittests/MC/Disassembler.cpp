//===- llvm/unittest/Object/Disassembler.cpp ------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm-c/Disassembler.h"
#include "llvm/Support/TargetSelect.h"
#include "gtest/gtest.h"

using namespace llvm;

static const char *symbolLookupCallback(void *DisInfo, uint64_t ReferenceValue,
                                        uint64_t *ReferenceType,
                                        uint64_t ReferencePC,
                                        const char **ReferenceName) {
  *ReferenceType = LLVMDisassembler_ReferenceType_InOut_None;
  return nullptr;
}

TEST(Disassembler, X86Test) {
  llvm::InitializeAllTargetInfos();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllDisassemblers();

  uint8_t Bytes[] = {0x90, 0x90, 0xeb, 0xfd};
  uint8_t *BytesP = Bytes;
  const char OutStringSize = 100;
  char OutString[OutStringSize];
  LLVMDisasmContextRef DCR = LLVMCreateDisasm("x86_64-pc-linux", nullptr, 0,
                                              nullptr, symbolLookupCallback);
  if (!DCR)
    return;

  size_t InstSize;
  unsigned NumBytes = sizeof(Bytes);
  unsigned PC = 0;

  InstSize = LLVMDisasmInstruction(DCR, BytesP, NumBytes, PC, OutString,
                                   OutStringSize);
  EXPECT_EQ(InstSize, 1U);
  EXPECT_EQ(StringRef(OutString), "\tnop");
  PC += InstSize;
  BytesP += InstSize;
  NumBytes -= InstSize;

  InstSize = LLVMDisasmInstruction(DCR, BytesP, NumBytes, PC, OutString,
                                   OutStringSize);
  EXPECT_EQ(InstSize, 1U);
  EXPECT_EQ(StringRef(OutString), "\tnop");
  PC += InstSize;
  BytesP += InstSize;
  NumBytes -= InstSize;

  InstSize = LLVMDisasmInstruction(DCR, BytesP, NumBytes, PC, OutString,
                                   OutStringSize);
  EXPECT_EQ(InstSize, 2U);
  EXPECT_EQ(StringRef(OutString), "\tjmp\t0x1");

  LLVMDisasmDispose(DCR);
}

TEST(Disassembler, WebAssemblyTest) {
  llvm::InitializeAllTargetInfos();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllDisassemblers();

  uint8_t Bytes[] = {0x6a, 0x42, 0x7F, 0x35, 0x01, 0x10};
  uint8_t *BytesP = Bytes;
  const char OutStringSize = 100;
  char OutString[OutStringSize];
  LLVMDisasmContextRef DCR = LLVMCreateDisasm(
      "wasm32-unknown-unknown-elf", nullptr, 0, nullptr, symbolLookupCallback);
  if (!DCR)
    return;

  size_t InstSize;
  unsigned NumBytes = sizeof(Bytes);
  unsigned PC = 0;

  InstSize = LLVMDisasmInstruction(DCR, BytesP, NumBytes, PC, OutString,
                                   OutStringSize);
  EXPECT_EQ(InstSize, 1U);
  EXPECT_EQ(StringRef(OutString), "\ti32.add ");
  PC += InstSize;
  BytesP += InstSize;
  NumBytes -= InstSize;

  InstSize = LLVMDisasmInstruction(DCR, BytesP, NumBytes, PC, OutString,
                                   OutStringSize);
  EXPECT_EQ(InstSize, 2U);
  EXPECT_EQ(StringRef(OutString), "\ti64.const\t-1");

  PC += InstSize;
  BytesP += InstSize;
  NumBytes -= InstSize;

  InstSize = LLVMDisasmInstruction(DCR, BytesP, NumBytes, PC, OutString,
                                   OutStringSize);
  EXPECT_EQ(InstSize, 3U);
  EXPECT_EQ(StringRef(OutString), "\ti64.load32_u\t16, :p2align=1");

  LLVMDisasmDispose(DCR);
}
