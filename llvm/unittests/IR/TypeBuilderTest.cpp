//===- llvm/unittest/TypeBuilderTest.cpp - TypeBuilder tests --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/TypeBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

TEST(TypeBuilderTest, Void) {
  LLVMContext Context;
  EXPECT_EQ(Type::getVoidTy(Context), (TypeBuilder<void, true>::get(Context)));
  EXPECT_EQ(Type::getVoidTy(Context), (TypeBuilder<void, false>::get(Context)));
  // Special cases for C compatibility:
  EXPECT_EQ(Type::getInt8PtrTy(Context),
            (TypeBuilder<void *, false>::get(Context)));
  EXPECT_EQ(Type::getInt8PtrTy(Context),
            (TypeBuilder<const void *, false>::get(Context)));
  EXPECT_EQ(Type::getInt8PtrTy(Context),
            (TypeBuilder<volatile void *, false>::get(Context)));
  EXPECT_EQ(Type::getInt8PtrTy(Context),
            (TypeBuilder<const volatile void *, false>::get(Context)));
}

TEST(TypeBuilderTest, HostIntegers) {
  LLVMContext Context;
  EXPECT_EQ(Type::getInt8Ty(Context),
            (TypeBuilder<int8_t, false>::get(Context)));
  EXPECT_EQ(Type::getInt8Ty(Context),
            (TypeBuilder<uint8_t, false>::get(Context)));
  EXPECT_EQ(Type::getInt16Ty(Context),
            (TypeBuilder<int16_t, false>::get(Context)));
  EXPECT_EQ(Type::getInt16Ty(Context),
            (TypeBuilder<uint16_t, false>::get(Context)));
  EXPECT_EQ(Type::getInt32Ty(Context),
            (TypeBuilder<int32_t, false>::get(Context)));
  EXPECT_EQ(Type::getInt32Ty(Context),
            (TypeBuilder<uint32_t, false>::get(Context)));
  EXPECT_EQ(Type::getInt64Ty(Context),
            (TypeBuilder<int64_t, false>::get(Context)));
  EXPECT_EQ(Type::getInt64Ty(Context),
            (TypeBuilder<uint64_t, false>::get(Context)));

  EXPECT_EQ(IntegerType::get(Context, sizeof(size_t) * CHAR_BIT),
            (TypeBuilder<size_t, false>::get(Context)));
  EXPECT_EQ(IntegerType::get(Context, sizeof(ptrdiff_t) * CHAR_BIT),
            (TypeBuilder<ptrdiff_t, false>::get(Context)));
}

TEST(TypeBuilderTest, CrossCompilableIntegers) {
  LLVMContext Context;
  EXPECT_EQ(IntegerType::get(Context, 1),
            (TypeBuilder<types::i<1>, true>::get(Context)));
  EXPECT_EQ(IntegerType::get(Context, 1),
            (TypeBuilder<types::i<1>, false>::get(Context)));
  EXPECT_EQ(IntegerType::get(Context, 72),
            (TypeBuilder<types::i<72>, true>::get(Context)));
  EXPECT_EQ(IntegerType::get(Context, 72),
            (TypeBuilder<types::i<72>, false>::get(Context)));
}

TEST(TypeBuilderTest, Float) {
  LLVMContext Context;
  EXPECT_EQ(Type::getFloatTy(Context),
            (TypeBuilder<float, false>::get(Context)));
  EXPECT_EQ(Type::getDoubleTy(Context),
            (TypeBuilder<double, false>::get(Context)));
  // long double isn't supported yet.
  EXPECT_EQ(Type::getFloatTy(Context),
            (TypeBuilder<types::ieee_float, true>::get(Context)));
  EXPECT_EQ(Type::getFloatTy(Context),
            (TypeBuilder<types::ieee_float, false>::get(Context)));
  EXPECT_EQ(Type::getDoubleTy(Context),
            (TypeBuilder<types::ieee_double, true>::get(Context)));
  EXPECT_EQ(Type::getDoubleTy(Context),
            (TypeBuilder<types::ieee_double, false>::get(Context)));
  EXPECT_EQ(Type::getX86_FP80Ty(Context),
            (TypeBuilder<types::x86_fp80, true>::get(Context)));
  EXPECT_EQ(Type::getX86_FP80Ty(Context),
            (TypeBuilder<types::x86_fp80, false>::get(Context)));
  EXPECT_EQ(Type::getFP128Ty(Context),
            (TypeBuilder<types::fp128, true>::get(Context)));
  EXPECT_EQ(Type::getFP128Ty(Context),
            (TypeBuilder<types::fp128, false>::get(Context)));
  EXPECT_EQ(Type::getPPC_FP128Ty(Context),
            (TypeBuilder<types::ppc_fp128, true>::get(Context)));
  EXPECT_EQ(Type::getPPC_FP128Ty(Context),
            (TypeBuilder<types::ppc_fp128, false>::get(Context)));
}

TEST(TypeBuilderTest, Derived) {
  LLVMContext Context;
  EXPECT_EQ(PointerType::getUnqual(Type::getInt8PtrTy(Context)),
            (TypeBuilder<int8_t **, false>::get(Context)));
  EXPECT_EQ(ArrayType::get(Type::getInt8Ty(Context), 7),
            (TypeBuilder<int8_t[7], false>::get(Context)));
  EXPECT_EQ(ArrayType::get(Type::getInt8Ty(Context), 0),
            (TypeBuilder<int8_t[], false>::get(Context)));

  EXPECT_EQ(PointerType::getUnqual(Type::getInt8PtrTy(Context)),
            (TypeBuilder<types::i<8> **, false>::get(Context)));
  EXPECT_EQ(ArrayType::get(Type::getInt8Ty(Context), 7),
            (TypeBuilder<types::i<8>[7], false>::get(Context)));
  EXPECT_EQ(ArrayType::get(Type::getInt8Ty(Context), 0),
            (TypeBuilder<types::i<8>[], false>::get(Context)));

  EXPECT_EQ(PointerType::getUnqual(Type::getInt8PtrTy(Context)),
            (TypeBuilder<types::i<8> **, true>::get(Context)));
  EXPECT_EQ(ArrayType::get(Type::getInt8Ty(Context), 7),
            (TypeBuilder<types::i<8>[7], true>::get(Context)));
  EXPECT_EQ(ArrayType::get(Type::getInt8Ty(Context), 0),
            (TypeBuilder<types::i<8>[], true>::get(Context)));

  EXPECT_EQ(Type::getInt8Ty(Context),
            (TypeBuilder<const int8_t, false>::get(Context)));
  EXPECT_EQ(Type::getInt8Ty(Context),
            (TypeBuilder<volatile int8_t, false>::get(Context)));
  EXPECT_EQ(Type::getInt8Ty(Context),
            (TypeBuilder<const volatile int8_t, false>::get(Context)));

  EXPECT_EQ(Type::getInt8Ty(Context),
            (TypeBuilder<const types::i<8>, false>::get(Context)));
  EXPECT_EQ(Type::getInt8Ty(Context),
            (TypeBuilder<volatile types::i<8>, false>::get(Context)));
  EXPECT_EQ(Type::getInt8Ty(Context),
            (TypeBuilder<const volatile types::i<8>, false>::get(Context)));

  EXPECT_EQ(Type::getInt8Ty(Context),
            (TypeBuilder<const types::i<8>, true>::get(Context)));
  EXPECT_EQ(Type::getInt8Ty(Context),
            (TypeBuilder<volatile types::i<8>, true>::get(Context)));
  EXPECT_EQ(Type::getInt8Ty(Context),
            (TypeBuilder<const volatile types::i<8>, true>::get(Context)));

  EXPECT_EQ(Type::getInt8PtrTy(Context),
            (TypeBuilder<const volatile int8_t *const volatile, false>::get(
                Context)));
}

TEST(TypeBuilderTest, Functions) {
  LLVMContext Context;
  std::vector<Type*> params;
  EXPECT_EQ(FunctionType::get(Type::getVoidTy(Context), params, false),
            (TypeBuilder<void(), true>::get(Context)));
  EXPECT_EQ(FunctionType::get(Type::getInt8Ty(Context), params, true),
            (TypeBuilder<int8_t(...), false>::get(Context)));
  params.push_back(TypeBuilder<int32_t *, false>::get(Context));
  EXPECT_EQ(FunctionType::get(Type::getInt8Ty(Context), params, false),
            (TypeBuilder<int8_t(const int32_t *), false>::get(Context)));
  EXPECT_EQ(FunctionType::get(Type::getInt8Ty(Context), params, true),
            (TypeBuilder<int8_t(const int32_t *, ...), false>::get(Context)));
  params.push_back(TypeBuilder<char *, false>::get(Context));
  EXPECT_EQ(FunctionType::get(Type::getInt8Ty(Context), params, false),
            (TypeBuilder<int8_t(int32_t *, void *), false>::get(Context)));
  EXPECT_EQ(FunctionType::get(Type::getInt8Ty(Context), params, true),
            (TypeBuilder<int8_t(int32_t *, char *, ...), false>::get(Context)));
  params.push_back(TypeBuilder<char, false>::get(Context));
  EXPECT_EQ(
      FunctionType::get(Type::getInt8Ty(Context), params, false),
      (TypeBuilder<int8_t(int32_t *, void *, char), false>::get(Context)));
  EXPECT_EQ(
      FunctionType::get(Type::getInt8Ty(Context), params, true),
      (TypeBuilder<int8_t(int32_t *, char *, char, ...), false>::get(Context)));
  params.push_back(TypeBuilder<char, false>::get(Context));
  EXPECT_EQ(FunctionType::get(Type::getInt8Ty(Context), params, false),
            (TypeBuilder<int8_t(int32_t *, void *, char, char), false>::get(
                Context)));
  EXPECT_EQ(
      FunctionType::get(Type::getInt8Ty(Context), params, true),
      (TypeBuilder<int8_t(int32_t *, char *, char, char, ...), false>::get(
          Context)));
  params.push_back(TypeBuilder<char, false>::get(Context));
  EXPECT_EQ(
      FunctionType::get(Type::getInt8Ty(Context), params, false),
      (TypeBuilder<int8_t(int32_t *, void *, char, char, char), false>::get(
          Context)));
  EXPECT_EQ(FunctionType::get(Type::getInt8Ty(Context), params, true),
            (TypeBuilder<int8_t(int32_t *, char *, char, char, char, ...),
                         false>::get(Context)));
}

TEST(TypeBuilderTest, Context) {
  // We used to cache TypeBuilder results in static local variables.  This
  // produced the same type for different contexts, which of course broke
  // things.
  LLVMContext context1;
  EXPECT_EQ(&context1,
            &(TypeBuilder<types::i<1>, true>::get(context1))->getContext());
  LLVMContext context2;
  EXPECT_EQ(&context2,
            &(TypeBuilder<types::i<1>, true>::get(context2))->getContext());
}

struct MyType {
  int a;
  int *b;
  void *array[1];
};

struct MyPortableType {
  int32_t a;
  int32_t *b;
  void *array[1];
};

}  // anonymous namespace

namespace llvm {
template<bool cross> class TypeBuilder<MyType, cross> {
public:
  static StructType *get(LLVMContext &Context) {
    // Using the static result variable ensures that the type is
    // only looked up once.
    std::vector<Type*> st;
    st.push_back(TypeBuilder<int, cross>::get(Context));
    st.push_back(TypeBuilder<int*, cross>::get(Context));
    st.push_back(TypeBuilder<void*[], cross>::get(Context));
    static StructType *const result = StructType::get(Context, st);
    return result;
  }

  // You may find this a convenient place to put some constants
  // to help with getelementptr.  They don't have any effect on
  // the operation of TypeBuilder.
  enum Fields {
    FIELD_A,
    FIELD_B,
    FIELD_ARRAY
  };
};

template<bool cross> class TypeBuilder<MyPortableType, cross> {
public:
  static StructType *get(LLVMContext &Context) {
    // Using the static result variable ensures that the type is
    // only looked up once.
    std::vector<Type*> st;
    st.push_back(TypeBuilder<types::i<32>, cross>::get(Context));
    st.push_back(TypeBuilder<types::i<32>*, cross>::get(Context));
    st.push_back(TypeBuilder<types::i<8>*[], cross>::get(Context));
    static StructType *const result = StructType::get(Context, st);
    return result;
  }

  // You may find this a convenient place to put some constants
  // to help with getelementptr.  They don't have any effect on
  // the operation of TypeBuilder.
  enum Fields {
    FIELD_A,
    FIELD_B,
    FIELD_ARRAY
  };
};
}  // namespace llvm
namespace {

TEST(TypeBuilderTest, Extensions) {
  LLVMContext Context;
  EXPECT_EQ(PointerType::getUnqual(StructType::get(
                TypeBuilder<int, false>::get(Context),
                TypeBuilder<int *, false>::get(Context),
                TypeBuilder<void *[], false>::get(Context), (void *)nullptr)),
            (TypeBuilder<MyType *, false>::get(Context)));
  EXPECT_EQ(
      PointerType::getUnqual(StructType::get(
          TypeBuilder<types::i<32>, false>::get(Context),
          TypeBuilder<types::i<32> *, false>::get(Context),
          TypeBuilder<types::i<8> *[], false>::get(Context), (void *)nullptr)),
      (TypeBuilder<MyPortableType *, false>::get(Context)));
  EXPECT_EQ(
      PointerType::getUnqual(StructType::get(
          TypeBuilder<types::i<32>, false>::get(Context),
          TypeBuilder<types::i<32> *, false>::get(Context),
          TypeBuilder<types::i<8> *[], false>::get(Context), (void *)nullptr)),
      (TypeBuilder<MyPortableType *, true>::get(Context)));
}

}  // anonymous namespace
