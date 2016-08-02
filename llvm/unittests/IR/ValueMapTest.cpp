//===- llvm/unittest/ADT/ValueMapTest.cpp - ValueMap unit tests -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/ValueMap.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

// Test fixture
template<typename T>
class ValueMapTest : public testing::Test {
protected:
  Constant *ConstantV;
  std::unique_ptr<BitCastInst> BitcastV;
  std::unique_ptr<BinaryOperator> AddV;

  ValueMapTest() :
    ConstantV(ConstantInt::get(Type::getInt32Ty(getGlobalContext()), 0)),
    BitcastV(new BitCastInst(ConstantV, Type::getInt32Ty(getGlobalContext()))),
    AddV(BinaryOperator::CreateAdd(ConstantV, ConstantV)) {
  }
};

// Run everything on Value*, a subtype to make sure that casting works as
// expected, and a const subtype to make sure we cast const correctly.
typedef ::testing::Types<Value, Instruction, const Instruction> KeyTypes;
TYPED_TEST_CASE(ValueMapTest, KeyTypes);

TYPED_TEST(ValueMapTest, Null) {
  ValueMap<TypeParam*, int> VM1;
  VM1[nullptr] = 7;
  EXPECT_EQ(7, VM1.lookup(nullptr));
}

TYPED_TEST(ValueMapTest, FollowsValue) {
  ValueMap<TypeParam*, int> VM;
  VM[this->BitcastV.get()] = 7;
  EXPECT_EQ(7, VM.lookup(this->BitcastV.get()));
  EXPECT_EQ(0u, VM.count(this->AddV.get()));
  this->BitcastV->replaceAllUsesWith(this->AddV.get());
  EXPECT_EQ(7, VM.lookup(this->AddV.get()));
  EXPECT_EQ(0u, VM.count(this->BitcastV.get()));
  this->AddV.reset();
  EXPECT_EQ(0u, VM.count(this->AddV.get()));
  EXPECT_EQ(0u, VM.count(this->BitcastV.get()));
  EXPECT_EQ(0U, VM.size());
}

TYPED_TEST(ValueMapTest, OperationsWork) {
  ValueMap<TypeParam*, int> VM;
  ValueMap<TypeParam*, int> VM2(16);  (void)VM2;
  typename ValueMapConfig<TypeParam*>::ExtraData Data;
  ValueMap<TypeParam*, int> VM3(Data, 16);  (void)VM3;
  EXPECT_TRUE(VM.empty());

  VM[this->BitcastV.get()] = 7;

  // Find:
  typename ValueMap<TypeParam*, int>::iterator I =
    VM.find(this->BitcastV.get());
  ASSERT_TRUE(I != VM.end());
  EXPECT_EQ(this->BitcastV.get(), I->first);
  EXPECT_EQ(7, I->second);
  EXPECT_TRUE(VM.find(this->AddV.get()) == VM.end());

  // Const find:
  const ValueMap<TypeParam*, int> &CVM = VM;
  typename ValueMap<TypeParam*, int>::const_iterator CI =
    CVM.find(this->BitcastV.get());
  ASSERT_TRUE(CI != CVM.end());
  EXPECT_EQ(this->BitcastV.get(), CI->first);
  EXPECT_EQ(7, CI->second);
  EXPECT_TRUE(CVM.find(this->AddV.get()) == CVM.end());

  // Insert:
  std::pair<typename ValueMap<TypeParam*, int>::iterator, bool> InsertResult1 =
    VM.insert(std::make_pair(this->AddV.get(), 3));
  EXPECT_EQ(this->AddV.get(), InsertResult1.first->first);
  EXPECT_EQ(3, InsertResult1.first->second);
  EXPECT_TRUE(InsertResult1.second);
  EXPECT_EQ(1u, VM.count(this->AddV.get()));
  std::pair<typename ValueMap<TypeParam*, int>::iterator, bool> InsertResult2 =
    VM.insert(std::make_pair(this->AddV.get(), 5));
  EXPECT_EQ(this->AddV.get(), InsertResult2.first->first);
  EXPECT_EQ(3, InsertResult2.first->second);
  EXPECT_FALSE(InsertResult2.second);

  // Erase:
  VM.erase(InsertResult2.first);
  EXPECT_EQ(0U, VM.count(this->AddV.get()));
  EXPECT_EQ(1U, VM.count(this->BitcastV.get()));
  VM.erase(this->BitcastV.get());
  EXPECT_EQ(0U, VM.count(this->BitcastV.get()));
  EXPECT_EQ(0U, VM.size());

  // Range insert:
  SmallVector<std::pair<Instruction*, int>, 2> Elems;
  Elems.push_back(std::make_pair(this->AddV.get(), 1));
  Elems.push_back(std::make_pair(this->BitcastV.get(), 2));
  VM.insert(Elems.begin(), Elems.end());
  EXPECT_EQ(1, VM.lookup(this->AddV.get()));
  EXPECT_EQ(2, VM.lookup(this->BitcastV.get()));
}

template<typename ExpectedType, typename VarType>
void CompileAssertHasType(VarType) {
  static_assert(std::is_same<ExpectedType, VarType>::value,
                "Not the same type");
}

TYPED_TEST(ValueMapTest, Iteration) {
  ValueMap<TypeParam*, int> VM;
  VM[this->BitcastV.get()] = 2;
  VM[this->AddV.get()] = 3;
  size_t size = 0;
  for (typename ValueMap<TypeParam*, int>::iterator I = VM.begin(), E = VM.end();
       I != E; ++I) {
    ++size;
    std::pair<TypeParam*, int> value = *I; (void)value;
    CompileAssertHasType<TypeParam*>(I->first);
    if (I->second == 2) {
      EXPECT_EQ(this->BitcastV.get(), I->first);
      I->second = 5;
    } else if (I->second == 3) {
      EXPECT_EQ(this->AddV.get(), I->first);
      I->second = 6;
    } else {
      ADD_FAILURE() << "Iterated through an extra value.";
    }
  }
  EXPECT_EQ(2U, size);
  EXPECT_EQ(5, VM[this->BitcastV.get()]);
  EXPECT_EQ(6, VM[this->AddV.get()]);

  size = 0;
  // Cast to const ValueMap to avoid a bug in DenseMap's iterators.
  const ValueMap<TypeParam*, int>& CVM = VM;
  for (typename ValueMap<TypeParam*, int>::const_iterator I = CVM.begin(),
         E = CVM.end(); I != E; ++I) {
    ++size;
    std::pair<TypeParam*, int> value = *I;  (void)value;
    CompileAssertHasType<TypeParam*>(I->first);
    if (I->second == 5) {
      EXPECT_EQ(this->BitcastV.get(), I->first);
    } else if (I->second == 6) {
      EXPECT_EQ(this->AddV.get(), I->first);
    } else {
      ADD_FAILURE() << "Iterated through an extra value.";
    }
  }
  EXPECT_EQ(2U, size);
}

TYPED_TEST(ValueMapTest, DefaultCollisionBehavior) {
  // By default, we overwrite the old value with the replaced value.
  ValueMap<TypeParam*, int> VM;
  VM[this->BitcastV.get()] = 7;
  VM[this->AddV.get()] = 9;
  this->BitcastV->replaceAllUsesWith(this->AddV.get());
  EXPECT_EQ(0u, VM.count(this->BitcastV.get()));
  EXPECT_EQ(9, VM.lookup(this->AddV.get()));
}

TYPED_TEST(ValueMapTest, ConfiguredCollisionBehavior) {
  // TODO: Implement this when someone needs it.
}

template<typename KeyT, typename MutexT>
struct LockMutex : ValueMapConfig<KeyT, MutexT> {
  struct ExtraData {
    MutexT *M;
    bool *CalledRAUW;
    bool *CalledDeleted;
  };
  static void onRAUW(const ExtraData &Data, KeyT Old, KeyT New) {
    *Data.CalledRAUW = true;
    EXPECT_FALSE(Data.M->try_lock()) << "Mutex should already be locked.";
  }
  static void onDelete(const ExtraData &Data, KeyT Old) {
    *Data.CalledDeleted = true;
    EXPECT_FALSE(Data.M->try_lock()) << "Mutex should already be locked.";
  }
  static MutexT *getMutex(const ExtraData &Data) { return Data.M; }
};
// FIXME: These tests started failing on Windows.
#if LLVM_ENABLE_THREADS && !defined(LLVM_ON_WIN32)
TYPED_TEST(ValueMapTest, LocksMutex) {
  sys::Mutex M(false);  // Not recursive.
  bool CalledRAUW = false, CalledDeleted = false;
  typedef LockMutex<TypeParam*, sys::Mutex> ConfigType;
  typename ConfigType::ExtraData Data = {&M, &CalledRAUW, &CalledDeleted};
  ValueMap<TypeParam*, int, ConfigType> VM(Data);
  VM[this->BitcastV.get()] = 7;
  this->BitcastV->replaceAllUsesWith(this->AddV.get());
  this->AddV.reset();
  EXPECT_TRUE(CalledRAUW);
  EXPECT_TRUE(CalledDeleted);
}
#endif

template<typename KeyT>
struct NoFollow : ValueMapConfig<KeyT> {
  enum { FollowRAUW = false };
};

TYPED_TEST(ValueMapTest, NoFollowRAUW) {
  ValueMap<TypeParam*, int, NoFollow<TypeParam*> > VM;
  VM[this->BitcastV.get()] = 7;
  EXPECT_EQ(7, VM.lookup(this->BitcastV.get()));
  EXPECT_EQ(0u, VM.count(this->AddV.get()));
  this->BitcastV->replaceAllUsesWith(this->AddV.get());
  EXPECT_EQ(7, VM.lookup(this->BitcastV.get()));
  EXPECT_EQ(0, VM.lookup(this->AddV.get()));
  this->AddV.reset();
  EXPECT_EQ(7, VM.lookup(this->BitcastV.get()));
  EXPECT_EQ(0, VM.lookup(this->AddV.get()));
  this->BitcastV.reset();
  EXPECT_EQ(0, VM.lookup(this->BitcastV.get()));
  EXPECT_EQ(0, VM.lookup(this->AddV.get()));
  EXPECT_EQ(0U, VM.size());
}

template<typename KeyT>
struct CountOps : ValueMapConfig<KeyT> {
  struct ExtraData {
    int *Deletions;
    int *RAUWs;
  };

  static void onRAUW(const ExtraData &Data, KeyT Old, KeyT New) {
    ++*Data.RAUWs;
  }
  static void onDelete(const ExtraData &Data, KeyT Old) {
    ++*Data.Deletions;
  }
};

TYPED_TEST(ValueMapTest, CallsConfig) {
  int Deletions = 0, RAUWs = 0;
  typename CountOps<TypeParam*>::ExtraData Data = {&Deletions, &RAUWs};
  ValueMap<TypeParam*, int, CountOps<TypeParam*> > VM(Data);
  VM[this->BitcastV.get()] = 7;
  this->BitcastV->replaceAllUsesWith(this->AddV.get());
  EXPECT_EQ(0, Deletions);
  EXPECT_EQ(1, RAUWs);
  this->AddV.reset();
  EXPECT_EQ(1, Deletions);
  EXPECT_EQ(1, RAUWs);
  this->BitcastV.reset();
  EXPECT_EQ(1, Deletions);
  EXPECT_EQ(1, RAUWs);
}

template<typename KeyT>
struct ModifyingConfig : ValueMapConfig<KeyT> {
  // We'll put a pointer here back to the ValueMap this key is in, so
  // that we can modify it (and clobber *this) before the ValueMap
  // tries to do the same modification.  In previous versions of
  // ValueMap, that exploded.
  typedef ValueMap<KeyT, int, ModifyingConfig<KeyT> > **ExtraData;

  static void onRAUW(ExtraData Map, KeyT Old, KeyT New) {
    (*Map)->erase(Old);
  }
  static void onDelete(ExtraData Map, KeyT Old) {
    (*Map)->erase(Old);
  }
};
TYPED_TEST(ValueMapTest, SurvivesModificationByConfig) {
  ValueMap<TypeParam*, int, ModifyingConfig<TypeParam*> > *MapAddress;
  ValueMap<TypeParam*, int, ModifyingConfig<TypeParam*> > VM(&MapAddress);
  MapAddress = &VM;
  // Now the ModifyingConfig can modify the Map inside a callback.
  VM[this->BitcastV.get()] = 7;
  this->BitcastV->replaceAllUsesWith(this->AddV.get());
  EXPECT_EQ(0u, VM.count(this->BitcastV.get()));
  EXPECT_EQ(0u, VM.count(this->AddV.get()));
  VM[this->AddV.get()] = 7;
  this->AddV.reset();
  EXPECT_EQ(0u, VM.count(this->AddV.get()));
}

}
