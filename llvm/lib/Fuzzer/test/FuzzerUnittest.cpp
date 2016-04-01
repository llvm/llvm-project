// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.

#include "FuzzerInternal.h"
#include "gtest/gtest.h"
#include <set>

using namespace fuzzer;

// For now, have LLVMFuzzerTestOneInput just to make it link.
// Later we may want to make unittests that actually call LLVMFuzzerTestOneInput.
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  abort();
}

TEST(Fuzzer, CrossOver) {
  Random Rand(0);
  MutationDispatcher MD(Rand);
  Unit A({0, 1, 2}), B({5, 6, 7});
  Unit C;
  Unit Expected[] = {
       { 0 },
       { 0, 1 },
       { 0, 5 },
       { 0, 1, 2 },
       { 0, 1, 5 },
       { 0, 5, 1 },
       { 0, 5, 6 },
       { 0, 1, 2, 5 },
       { 0, 1, 5, 2 },
       { 0, 1, 5, 6 },
       { 0, 5, 1, 2 },
       { 0, 5, 1, 6 },
       { 0, 5, 6, 1 },
       { 0, 5, 6, 7 },
       { 0, 1, 2, 5, 6 },
       { 0, 1, 5, 2, 6 },
       { 0, 1, 5, 6, 2 },
       { 0, 1, 5, 6, 7 },
       { 0, 5, 1, 2, 6 },
       { 0, 5, 1, 6, 2 },
       { 0, 5, 1, 6, 7 },
       { 0, 5, 6, 1, 2 },
       { 0, 5, 6, 1, 7 },
       { 0, 5, 6, 7, 1 },
       { 0, 1, 2, 5, 6, 7 },
       { 0, 1, 5, 2, 6, 7 },
       { 0, 1, 5, 6, 2, 7 },
       { 0, 1, 5, 6, 7, 2 },
       { 0, 5, 1, 2, 6, 7 },
       { 0, 5, 1, 6, 2, 7 },
       { 0, 5, 1, 6, 7, 2 },
       { 0, 5, 6, 1, 2, 7 },
       { 0, 5, 6, 1, 7, 2 },
       { 0, 5, 6, 7, 1, 2 }
  };
  for (size_t Len = 1; Len < 8; Len++) {
    std::set<Unit> FoundUnits, ExpectedUnitsWitThisLength;
    for (int Iter = 0; Iter < 3000; Iter++) {
      C.resize(Len);
      size_t NewSize = MD.CrossOver(A.data(), A.size(), B.data(), B.size(),
                                    C.data(), C.size());
      C.resize(NewSize);
      FoundUnits.insert(C);
    }
    for (const Unit &U : Expected)
      if (U.size() <= Len)
        ExpectedUnitsWitThisLength.insert(U);
    EXPECT_EQ(ExpectedUnitsWitThisLength, FoundUnits);
  }
}

TEST(Fuzzer, Hash) {
  uint8_t A[] = {'a', 'b', 'c'};
  fuzzer::Unit U(A, A + sizeof(A));
  EXPECT_EQ("a9993e364706816aba3e25717850c26c9cd0d89d", fuzzer::Hash(U));
  U.push_back('d');
  EXPECT_EQ("81fe8bfe87576c3ecb22426f8e57847382917acf", fuzzer::Hash(U));
}

typedef size_t (MutationDispatcher::*Mutator)(uint8_t *Data, size_t Size,
                                              size_t MaxSize);

void TestEraseByte(Mutator M, int NumIter) {
  uint8_t REM0[8] = {0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77};
  uint8_t REM1[8] = {0x00, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77};
  uint8_t REM2[8] = {0x00, 0x11, 0x33, 0x44, 0x55, 0x66, 0x77};
  uint8_t REM3[8] = {0x00, 0x11, 0x22, 0x44, 0x55, 0x66, 0x77};
  uint8_t REM4[8] = {0x00, 0x11, 0x22, 0x33, 0x55, 0x66, 0x77};
  uint8_t REM5[8] = {0x00, 0x11, 0x22, 0x33, 0x44, 0x66, 0x77};
  uint8_t REM6[8] = {0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x77};
  uint8_t REM7[8] = {0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66};
  Random Rand(0);
  MutationDispatcher MD(Rand);
  int FoundMask = 0;
  for (int i = 0; i < NumIter; i++) {
    uint8_t T[8] = {0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77};
    size_t NewSize = (MD.*M)(T, sizeof(T), sizeof(T));
    if (NewSize == 7 && !memcmp(REM0, T, 7)) FoundMask |= 1 << 0;
    if (NewSize == 7 && !memcmp(REM1, T, 7)) FoundMask |= 1 << 1;
    if (NewSize == 7 && !memcmp(REM2, T, 7)) FoundMask |= 1 << 2;
    if (NewSize == 7 && !memcmp(REM3, T, 7)) FoundMask |= 1 << 3;
    if (NewSize == 7 && !memcmp(REM4, T, 7)) FoundMask |= 1 << 4;
    if (NewSize == 7 && !memcmp(REM5, T, 7)) FoundMask |= 1 << 5;
    if (NewSize == 7 && !memcmp(REM6, T, 7)) FoundMask |= 1 << 6;
    if (NewSize == 7 && !memcmp(REM7, T, 7)) FoundMask |= 1 << 7;
  }
  EXPECT_EQ(FoundMask, 255);
}

TEST(FuzzerMutate, EraseByte1) {
  TestEraseByte(&MutationDispatcher::Mutate_EraseByte, 100);
}
TEST(FuzzerMutate, EraseByte2) {
  TestEraseByte(&MutationDispatcher::Mutate, 1000);
}

void TestInsertByte(Mutator M, int NumIter) {
  Random Rand(0);
  MutationDispatcher MD(Rand);
  int FoundMask = 0;
  uint8_t INS0[8] = {0xF1, 0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66};
  uint8_t INS1[8] = {0x00, 0xF2, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66};
  uint8_t INS2[8] = {0x00, 0x11, 0xF3, 0x22, 0x33, 0x44, 0x55, 0x66};
  uint8_t INS3[8] = {0x00, 0x11, 0x22, 0xF4, 0x33, 0x44, 0x55, 0x66};
  uint8_t INS4[8] = {0x00, 0x11, 0x22, 0x33, 0xF5, 0x44, 0x55, 0x66};
  uint8_t INS5[8] = {0x00, 0x11, 0x22, 0x33, 0x44, 0xF6, 0x55, 0x66};
  uint8_t INS6[8] = {0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0xF7, 0x66};
  uint8_t INS7[8] = {0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0xF8};
  for (int i = 0; i < NumIter; i++) {
    uint8_t T[8] = {0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66};
    size_t NewSize = (MD.*M)(T, 7, 8);
    if (NewSize == 8 && !memcmp(INS0, T, 8)) FoundMask |= 1 << 0;
    if (NewSize == 8 && !memcmp(INS1, T, 8)) FoundMask |= 1 << 1;
    if (NewSize == 8 && !memcmp(INS2, T, 8)) FoundMask |= 1 << 2;
    if (NewSize == 8 && !memcmp(INS3, T, 8)) FoundMask |= 1 << 3;
    if (NewSize == 8 && !memcmp(INS4, T, 8)) FoundMask |= 1 << 4;
    if (NewSize == 8 && !memcmp(INS5, T, 8)) FoundMask |= 1 << 5;
    if (NewSize == 8 && !memcmp(INS6, T, 8)) FoundMask |= 1 << 6;
    if (NewSize == 8 && !memcmp(INS7, T, 8)) FoundMask |= 1 << 7;
  }
  EXPECT_EQ(FoundMask, 255);
}

TEST(FuzzerMutate, InsertByte1) {
  TestInsertByte(&MutationDispatcher::Mutate_InsertByte, 1 << 15);
}
TEST(FuzzerMutate, InsertByte2) {
  TestInsertByte(&MutationDispatcher::Mutate, 1 << 17);
}

void TestChangeByte(Mutator M, int NumIter) {
  Random Rand(0);
  MutationDispatcher MD(Rand);
  int FoundMask = 0;
  uint8_t CH0[8] = {0xF0, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77};
  uint8_t CH1[8] = {0x00, 0xF1, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77};
  uint8_t CH2[8] = {0x00, 0x11, 0xF2, 0x33, 0x44, 0x55, 0x66, 0x77};
  uint8_t CH3[8] = {0x00, 0x11, 0x22, 0xF3, 0x44, 0x55, 0x66, 0x77};
  uint8_t CH4[8] = {0x00, 0x11, 0x22, 0x33, 0xF4, 0x55, 0x66, 0x77};
  uint8_t CH5[8] = {0x00, 0x11, 0x22, 0x33, 0x44, 0xF5, 0x66, 0x77};
  uint8_t CH6[8] = {0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0xF5, 0x77};
  uint8_t CH7[8] = {0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0xF7};
  for (int i = 0; i < NumIter; i++) {
    uint8_t T[9] = {0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77};
    size_t NewSize = (MD.*M)(T, 8, 9);
    if (NewSize == 8 && !memcmp(CH0, T, 8)) FoundMask |= 1 << 0;
    if (NewSize == 8 && !memcmp(CH1, T, 8)) FoundMask |= 1 << 1;
    if (NewSize == 8 && !memcmp(CH2, T, 8)) FoundMask |= 1 << 2;
    if (NewSize == 8 && !memcmp(CH3, T, 8)) FoundMask |= 1 << 3;
    if (NewSize == 8 && !memcmp(CH4, T, 8)) FoundMask |= 1 << 4;
    if (NewSize == 8 && !memcmp(CH5, T, 8)) FoundMask |= 1 << 5;
    if (NewSize == 8 && !memcmp(CH6, T, 8)) FoundMask |= 1 << 6;
    if (NewSize == 8 && !memcmp(CH7, T, 8)) FoundMask |= 1 << 7;
  }
  EXPECT_EQ(FoundMask, 255);
}

TEST(FuzzerMutate, ChangeByte1) {
  TestChangeByte(&MutationDispatcher::Mutate_ChangeByte, 1 << 15);
}
TEST(FuzzerMutate, ChangeByte2) {
  TestChangeByte(&MutationDispatcher::Mutate, 1 << 17);
}

void TestChangeBit(Mutator M, int NumIter) {
  Random Rand(0);
  MutationDispatcher MD(Rand);
  int FoundMask = 0;
  uint8_t CH0[8] = {0x01, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77};
  uint8_t CH1[8] = {0x00, 0x13, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77};
  uint8_t CH2[8] = {0x00, 0x11, 0x02, 0x33, 0x44, 0x55, 0x66, 0x77};
  uint8_t CH3[8] = {0x00, 0x11, 0x22, 0x37, 0x44, 0x55, 0x66, 0x77};
  uint8_t CH4[8] = {0x00, 0x11, 0x22, 0x33, 0x54, 0x55, 0x66, 0x77};
  uint8_t CH5[8] = {0x00, 0x11, 0x22, 0x33, 0x44, 0x54, 0x66, 0x77};
  uint8_t CH6[8] = {0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x76, 0x77};
  uint8_t CH7[8] = {0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0xF7};
  for (int i = 0; i < NumIter; i++) {
    uint8_t T[9] = {0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77};
    size_t NewSize = (MD.*M)(T, 8, 9);
    if (NewSize == 8 && !memcmp(CH0, T, 8)) FoundMask |= 1 << 0;
    if (NewSize == 8 && !memcmp(CH1, T, 8)) FoundMask |= 1 << 1;
    if (NewSize == 8 && !memcmp(CH2, T, 8)) FoundMask |= 1 << 2;
    if (NewSize == 8 && !memcmp(CH3, T, 8)) FoundMask |= 1 << 3;
    if (NewSize == 8 && !memcmp(CH4, T, 8)) FoundMask |= 1 << 4;
    if (NewSize == 8 && !memcmp(CH5, T, 8)) FoundMask |= 1 << 5;
    if (NewSize == 8 && !memcmp(CH6, T, 8)) FoundMask |= 1 << 6;
    if (NewSize == 8 && !memcmp(CH7, T, 8)) FoundMask |= 1 << 7;
  }
  EXPECT_EQ(FoundMask, 255);
}

TEST(FuzzerMutate, ChangeBit1) {
  TestChangeBit(&MutationDispatcher::Mutate_ChangeBit, 1 << 16);
}
TEST(FuzzerMutate, ChangeBit2) {
  TestChangeBit(&MutationDispatcher::Mutate, 1 << 18);
}

void TestShuffleBytes(Mutator M, int NumIter) {
  Random Rand(0);
  MutationDispatcher MD(Rand);
  int FoundMask = 0;
  uint8_t CH0[7] = {0x00, 0x22, 0x11, 0x33, 0x44, 0x55, 0x66};
  uint8_t CH1[7] = {0x11, 0x00, 0x33, 0x22, 0x44, 0x55, 0x66};
  uint8_t CH2[7] = {0x00, 0x33, 0x11, 0x22, 0x44, 0x55, 0x66};
  uint8_t CH3[7] = {0x00, 0x11, 0x22, 0x44, 0x55, 0x66, 0x33};
  uint8_t CH4[7] = {0x00, 0x11, 0x22, 0x33, 0x55, 0x44, 0x66};
  for (int i = 0; i < NumIter; i++) {
    uint8_t T[7] = {0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66};
    size_t NewSize = (MD.*M)(T, 7, 7);
    if (NewSize == 7 && !memcmp(CH0, T, 7)) FoundMask |= 1 << 0;
    if (NewSize == 7 && !memcmp(CH1, T, 7)) FoundMask |= 1 << 1;
    if (NewSize == 7 && !memcmp(CH2, T, 7)) FoundMask |= 1 << 2;
    if (NewSize == 7 && !memcmp(CH3, T, 7)) FoundMask |= 1 << 3;
    if (NewSize == 7 && !memcmp(CH4, T, 7)) FoundMask |= 1 << 4;
  }
  EXPECT_EQ(FoundMask, 31);
}

TEST(FuzzerMutate, ShuffleBytes1) {
  TestShuffleBytes(&MutationDispatcher::Mutate_ShuffleBytes, 1 << 16);
}
TEST(FuzzerMutate, ShuffleBytes2) {
  TestShuffleBytes(&MutationDispatcher::Mutate, 1 << 19);
}

void TestAddWordFromDictionary(Mutator M, int NumIter) {
  Random Rand(0);
  MutationDispatcher MD(Rand);
  uint8_t Word1[4] = {0xAA, 0xBB, 0xCC, 0xDD};
  uint8_t Word2[3] = {0xFF, 0xEE, 0xEF};
  MD.AddWordToManualDictionary(Word(Word1, sizeof(Word1)));
  MD.AddWordToManualDictionary(Word(Word2, sizeof(Word2)));
  int FoundMask = 0;
  uint8_t CH0[7] = {0x00, 0x11, 0x22, 0xAA, 0xBB, 0xCC, 0xDD};
  uint8_t CH1[7] = {0x00, 0x11, 0xAA, 0xBB, 0xCC, 0xDD, 0x22};
  uint8_t CH2[7] = {0x00, 0xAA, 0xBB, 0xCC, 0xDD, 0x11, 0x22};
  uint8_t CH3[7] = {0xAA, 0xBB, 0xCC, 0xDD, 0x00, 0x11, 0x22};
  uint8_t CH4[6] = {0x00, 0x11, 0x22, 0xFF, 0xEE, 0xEF};
  uint8_t CH5[6] = {0x00, 0x11, 0xFF, 0xEE, 0xEF, 0x22};
  uint8_t CH6[6] = {0x00, 0xFF, 0xEE, 0xEF, 0x11, 0x22};
  uint8_t CH7[6] = {0xFF, 0xEE, 0xEF, 0x00, 0x11, 0x22};
  for (int i = 0; i < NumIter; i++) {
    uint8_t T[7] = {0x00, 0x11, 0x22};
    size_t NewSize = (MD.*M)(T, 3, 7);
    if (NewSize == 7 && !memcmp(CH0, T, 7)) FoundMask |= 1 << 0;
    if (NewSize == 7 && !memcmp(CH1, T, 7)) FoundMask |= 1 << 1;
    if (NewSize == 7 && !memcmp(CH2, T, 7)) FoundMask |= 1 << 2;
    if (NewSize == 7 && !memcmp(CH3, T, 7)) FoundMask |= 1 << 3;
    if (NewSize == 6 && !memcmp(CH4, T, 6)) FoundMask |= 1 << 4;
    if (NewSize == 6 && !memcmp(CH5, T, 6)) FoundMask |= 1 << 5;
    if (NewSize == 6 && !memcmp(CH6, T, 6)) FoundMask |= 1 << 6;
    if (NewSize == 6 && !memcmp(CH7, T, 6)) FoundMask |= 1 << 7;
  }
  EXPECT_EQ(FoundMask, 255);
}

TEST(FuzzerMutate, AddWordFromDictionary1) {
  TestAddWordFromDictionary(
      &MutationDispatcher::Mutate_AddWordFromManualDictionary, 1 << 15);
}

TEST(FuzzerMutate, AddWordFromDictionary2) {
  TestAddWordFromDictionary(&MutationDispatcher::Mutate, 1 << 15);
}

void TestAddWordFromDictionaryWithHint(Mutator M, int NumIter) {
  Random Rand(0);
  MutationDispatcher MD(Rand);
  uint8_t W[] = {0xAA, 0xBB, 0xCC, 0xDD, 0xFF, 0xEE, 0xEF};
  size_t PosHint = 7777;
  MD.AddWordToAutoDictionary(Word(W, sizeof(W)), PosHint);
  int FoundMask = 0;
  for (int i = 0; i < NumIter; i++) {
    uint8_t T[10000];
    memset(T, 0, sizeof(T));
    size_t NewSize = (MD.*M)(T, 9000, 10000);
    if (NewSize >= PosHint + sizeof(W) &&
        !memcmp(W, T + PosHint, sizeof(W)))
      FoundMask = 1;
  }
  EXPECT_EQ(FoundMask, 1);
}

TEST(FuzzerMutate, AddWordFromDictionaryWithHint1) {
  TestAddWordFromDictionaryWithHint(
      &MutationDispatcher::Mutate_AddWordFromTemporaryAutoDictionary, 1 << 5);
}

TEST(FuzzerMutate, AddWordFromDictionaryWithHint2) {
  TestAddWordFromDictionaryWithHint(&MutationDispatcher::Mutate, 1 << 10);
}

void TestChangeASCIIInteger(Mutator M, int NumIter) {
  Random Rand(0);
  MutationDispatcher MD(Rand);

  uint8_t CH0[8] = {'1', '2', '3', '4', '5', '6', '7', '7'};
  uint8_t CH1[8] = {'1', '2', '3', '4', '5', '6', '7', '9'};
  uint8_t CH2[8] = {'2', '4', '6', '9', '1', '3', '5', '6'};
  uint8_t CH3[8] = {'0', '6', '1', '7', '2', '8', '3', '9'};
  int FoundMask = 0;
  for (int i = 0; i < NumIter; i++) {
    uint8_t T[8] = {'1', '2', '3', '4', '5', '6', '7', '8'};
    size_t NewSize = (MD.*M)(T, 8, 8);
    /**/ if (NewSize == 8 && !memcmp(CH0, T, 8)) FoundMask |= 1 << 0;
    else if (NewSize == 8 && !memcmp(CH1, T, 8)) FoundMask |= 1 << 1;
    else if (NewSize == 8 && !memcmp(CH2, T, 8)) FoundMask |= 1 << 2;
    else if (NewSize == 8 && !memcmp(CH3, T, 8)) FoundMask |= 1 << 3;
    else if (NewSize == 8)                       FoundMask |= 1 << 4;
  }
  EXPECT_EQ(FoundMask, 31);
}

TEST(FuzzerMutate, ChangeASCIIInteger1) {
  TestChangeASCIIInteger(&MutationDispatcher::Mutate_ChangeASCIIInteger,
                         1 << 15);
}

TEST(FuzzerMutate, ChangeASCIIInteger2) {
  TestChangeASCIIInteger(&MutationDispatcher::Mutate, 1 << 15);
}


TEST(FuzzerDictionary, ParseOneDictionaryEntry) {
  Unit U;
  EXPECT_FALSE(ParseOneDictionaryEntry("", &U));
  EXPECT_FALSE(ParseOneDictionaryEntry(" ", &U));
  EXPECT_FALSE(ParseOneDictionaryEntry("\t  ", &U));
  EXPECT_FALSE(ParseOneDictionaryEntry("  \" ", &U));
  EXPECT_FALSE(ParseOneDictionaryEntry("  zz\" ", &U));
  EXPECT_FALSE(ParseOneDictionaryEntry("  \"zz ", &U));
  EXPECT_FALSE(ParseOneDictionaryEntry("  \"\" ", &U));
  EXPECT_TRUE(ParseOneDictionaryEntry("\"a\"", &U));
  EXPECT_EQ(U, Unit({'a'}));
  EXPECT_TRUE(ParseOneDictionaryEntry("\"abc\"", &U));
  EXPECT_EQ(U, Unit({'a', 'b', 'c'}));
  EXPECT_TRUE(ParseOneDictionaryEntry("abc=\"abc\"", &U));
  EXPECT_EQ(U, Unit({'a', 'b', 'c'}));
  EXPECT_FALSE(ParseOneDictionaryEntry("\"\\\"", &U));
  EXPECT_TRUE(ParseOneDictionaryEntry("\"\\\\\"", &U));
  EXPECT_EQ(U, Unit({'\\'}));
  EXPECT_TRUE(ParseOneDictionaryEntry("\"\\xAB\"", &U));
  EXPECT_EQ(U, Unit({0xAB}));
  EXPECT_TRUE(ParseOneDictionaryEntry("\"\\xABz\\xDE\"", &U));
  EXPECT_EQ(U, Unit({0xAB, 'z', 0xDE}));
  EXPECT_TRUE(ParseOneDictionaryEntry("\"#\"", &U));
  EXPECT_EQ(U, Unit({'#'}));
  EXPECT_TRUE(ParseOneDictionaryEntry("\"\\\"\"", &U));
  EXPECT_EQ(U, Unit({'"'}));
}

TEST(FuzzerDictionary, ParseDictionaryFile) {
  std::vector<Unit> Units;
  EXPECT_FALSE(ParseDictionaryFile("zzz\n", &Units));
  EXPECT_FALSE(ParseDictionaryFile("", &Units));
  EXPECT_TRUE(ParseDictionaryFile("\n", &Units));
  EXPECT_EQ(Units.size(), 0U);
  EXPECT_TRUE(ParseDictionaryFile("#zzzz a b c d\n", &Units));
  EXPECT_EQ(Units.size(), 0U);
  EXPECT_TRUE(ParseDictionaryFile(" #zzzz\n", &Units));
  EXPECT_EQ(Units.size(), 0U);
  EXPECT_TRUE(ParseDictionaryFile("  #zzzz\n", &Units));
  EXPECT_EQ(Units.size(), 0U);
  EXPECT_TRUE(ParseDictionaryFile("  #zzzz\naaa=\"aa\"", &Units));
  EXPECT_EQ(Units, std::vector<Unit>({Unit({'a', 'a'})}));
  EXPECT_TRUE(
      ParseDictionaryFile("  #zzzz\naaa=\"aa\"\n\nabc=\"abc\"", &Units));
  EXPECT_EQ(Units,
            std::vector<Unit>({Unit({'a', 'a'}), Unit({'a', 'b', 'c'})}));
}

TEST(FuzzerUtil, Base64) {
  EXPECT_EQ("", Base64({}));
  EXPECT_EQ("YQ==", Base64({'a'}));
  EXPECT_EQ("eA==", Base64({'x'}));
  EXPECT_EQ("YWI=", Base64({'a', 'b'}));
  EXPECT_EQ("eHk=", Base64({'x', 'y'}));
  EXPECT_EQ("YWJj", Base64({'a', 'b', 'c'}));
  EXPECT_EQ("eHl6", Base64({'x', 'y', 'z'}));
  EXPECT_EQ("YWJjeA==", Base64({'a', 'b', 'c', 'x'}));
  EXPECT_EQ("YWJjeHk=", Base64({'a', 'b', 'c', 'x', 'y'}));
  EXPECT_EQ("YWJjeHl6", Base64({'a', 'b', 'c', 'x', 'y', 'z'}));
}

TEST(Corpus, Distribution) {
  Random Rand(0);
  MutationDispatcher MD(Rand);
  Fuzzer::FuzzingOptions Options;
  Fuzzer Fuzz(LLVMFuzzerTestOneInput, MD, Options);
  size_t N = 10;
  size_t TriesPerUnit = 1<<20;
  for (size_t i = 0; i < N; i++) {
    Fuzz.AddToCorpus(Unit{ static_cast<uint8_t>(i) });
  }
  std::vector<size_t> Hist(N);
  for (size_t i = 0; i < N * TriesPerUnit; i++) {
    Hist[Fuzz.ChooseUnitIdxToMutate()]++;
  }
  for (size_t i = 0; i < N; i++) {
    // A weak sanity check that every unit gets invoked.
    EXPECT_GT(Hist[i], TriesPerUnit / N / 3);
  }
}
