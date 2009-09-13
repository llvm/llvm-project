//===- llvm/unittest/Support/TypeBuilderTest.cpp - TypeBuilder tests -----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/TypeBuilder.h"
#include "llvm/LLVMContext.h"

#include "gtest/gtest.h"

using namespace llvm;

namespace {

TEST(TypeBuilderTest, Void) {
  EXPECT_EQ(Type::getVoidTy(getGlobalContext()), (TypeBuilder<void, true>::get(getGlobalContext())));
  EXPECT_EQ(Type::getVoidTy(getGlobalContext()), (TypeBuilder<void, false>::get(getGlobalContext())));
  // Special case for C compatibility:
  EXPECT_EQ(PointerType::getUnqual(Type::getInt8Ty(getGlobalContext())),
            (TypeBuilder<void*, false>::get(getGlobalContext())));
}

TEST(TypeBuilderTest, HostIntegers) {
  EXPECT_EQ(Type::getInt8Ty(getGlobalContext()), (TypeBuilder<int8_t, false>::get(getGlobalContext())));
  EXPECT_EQ(Type::getInt8Ty(getGlobalContext()), (TypeBuilder<uint8_t, false>::get(getGlobalContext())));
  EXPECT_EQ(Type::getInt16Ty(getGlobalContext()), (TypeBuilder<int16_t, false>::get(getGlobalContext())));
  EXPECT_EQ(Type::getInt16Ty(getGlobalContext()), (TypeBuilder<uint16_t, false>::get(getGlobalContext())));
  EXPECT_EQ(Type::getInt32Ty(getGlobalContext()), (TypeBuilder<int32_t, false>::get(getGlobalContext())));
  EXPECT_EQ(Type::getInt32Ty(getGlobalContext()), (TypeBuilder<uint32_t, false>::get(getGlobalContext())));
  EXPECT_EQ(Type::getInt64Ty(getGlobalContext()), (TypeBuilder<int64_t, false>::get(getGlobalContext())));
  EXPECT_EQ(Type::getInt64Ty(getGlobalContext()), (TypeBuilder<uint64_t, false>::get(getGlobalContext())));

  EXPECT_EQ(IntegerType::get(getGlobalContext(), sizeof(size_t) * CHAR_BIT),
            (TypeBuilder<size_t, false>::get(getGlobalContext())));
  EXPECT_EQ(IntegerType::get(getGlobalContext(), sizeof(ptrdiff_t) * CHAR_BIT),
            (TypeBuilder<ptrdiff_t, false>::get(getGlobalContext())));
}

TEST(TypeBuilderTest, CrossCompilableIntegers) {
  EXPECT_EQ(IntegerType::get(getGlobalContext(), 1), (TypeBuilder<types::i<1>, true>::get(getGlobalContext())));
  EXPECT_EQ(IntegerType::get(getGlobalContext(), 1), (TypeBuilder<types::i<1>, false>::get(getGlobalContext())));
  EXPECT_EQ(IntegerType::get(getGlobalContext(), 72), (TypeBuilder<types::i<72>, true>::get(getGlobalContext())));
  EXPECT_EQ(IntegerType::get(getGlobalContext(), 72), (TypeBuilder<types::i<72>, false>::get(getGlobalContext())));
}

TEST(TypeBuilderTest, Float) {
  EXPECT_EQ(Type::getFloatTy(getGlobalContext()), (TypeBuilder<float, false>::get(getGlobalContext())));
  EXPECT_EQ(Type::getDoubleTy(getGlobalContext()), (TypeBuilder<double, false>::get(getGlobalContext())));
  // long double isn't supported yet.
  EXPECT_EQ(Type::getFloatTy(getGlobalContext()), (TypeBuilder<types::ieee_float, true>::get(getGlobalContext())));
  EXPECT_EQ(Type::getFloatTy(getGlobalContext()), (TypeBuilder<types::ieee_float, false>::get(getGlobalContext())));
  EXPECT_EQ(Type::getDoubleTy(getGlobalContext()), (TypeBuilder<types::ieee_double, true>::get(getGlobalContext())));
  EXPECT_EQ(Type::getDoubleTy(getGlobalContext()), (TypeBuilder<types::ieee_double, false>::get(getGlobalContext())));
  EXPECT_EQ(Type::getX86_FP80Ty(getGlobalContext()), (TypeBuilder<types::x86_fp80, true>::get(getGlobalContext())));
  EXPECT_EQ(Type::getX86_FP80Ty(getGlobalContext()), (TypeBuilder<types::x86_fp80, false>::get(getGlobalContext())));
  EXPECT_EQ(Type::getFP128Ty(getGlobalContext()), (TypeBuilder<types::fp128, true>::get(getGlobalContext())));
  EXPECT_EQ(Type::getFP128Ty(getGlobalContext()), (TypeBuilder<types::fp128, false>::get(getGlobalContext())));
  EXPECT_EQ(Type::getPPC_FP128Ty(getGlobalContext()), (TypeBuilder<types::ppc_fp128, true>::get(getGlobalContext())));
  EXPECT_EQ(Type::getPPC_FP128Ty(getGlobalContext()), (TypeBuilder<types::ppc_fp128, false>::get(getGlobalContext())));
}

TEST(TypeBuilderTest, Derived) {
  EXPECT_EQ(PointerType::getUnqual(PointerType::getUnqual(Type::getInt8Ty(getGlobalContext()))),
            (TypeBuilder<int8_t**, false>::get(getGlobalContext())));
  EXPECT_EQ(ArrayType::get(Type::getInt8Ty(getGlobalContext()), 7),
            (TypeBuilder<int8_t[7], false>::get(getGlobalContext())));
  EXPECT_EQ(ArrayType::get(Type::getInt8Ty(getGlobalContext()), 0),
            (TypeBuilder<int8_t[], false>::get(getGlobalContext())));

  EXPECT_EQ(PointerType::getUnqual(PointerType::getUnqual(Type::getInt8Ty(getGlobalContext()))),
            (TypeBuilder<types::i<8>**, false>::get(getGlobalContext())));
  EXPECT_EQ(ArrayType::get(Type::getInt8Ty(getGlobalContext()), 7),
            (TypeBuilder<types::i<8>[7], false>::get(getGlobalContext())));
  EXPECT_EQ(ArrayType::get(Type::getInt8Ty(getGlobalContext()), 0),
            (TypeBuilder<types::i<8>[], false>::get(getGlobalContext())));

  EXPECT_EQ(PointerType::getUnqual(PointerType::getUnqual(Type::getInt8Ty(getGlobalContext()))),
            (TypeBuilder<types::i<8>**, true>::get(getGlobalContext())));
  EXPECT_EQ(ArrayType::get(Type::getInt8Ty(getGlobalContext()), 7),
            (TypeBuilder<types::i<8>[7], true>::get(getGlobalContext())));
  EXPECT_EQ(ArrayType::get(Type::getInt8Ty(getGlobalContext()), 0),
            (TypeBuilder<types::i<8>[], true>::get(getGlobalContext())));


  EXPECT_EQ(Type::getInt8Ty(getGlobalContext()),
            (TypeBuilder<const int8_t, false>::get(getGlobalContext())));
  EXPECT_EQ(Type::getInt8Ty(getGlobalContext()),
            (TypeBuilder<volatile int8_t, false>::get(getGlobalContext())));
  EXPECT_EQ(Type::getInt8Ty(getGlobalContext()),
            (TypeBuilder<const volatile int8_t, false>::get(getGlobalContext())));

  EXPECT_EQ(Type::getInt8Ty(getGlobalContext()),
            (TypeBuilder<const types::i<8>, false>::get(getGlobalContext())));
  EXPECT_EQ(Type::getInt8Ty(getGlobalContext()),
            (TypeBuilder<volatile types::i<8>, false>::get(getGlobalContext())));
  EXPECT_EQ(Type::getInt8Ty(getGlobalContext()),
            (TypeBuilder<const volatile types::i<8>, false>::get(getGlobalContext())));

  EXPECT_EQ(Type::getInt8Ty(getGlobalContext()),
            (TypeBuilder<const types::i<8>, true>::get(getGlobalContext())));
  EXPECT_EQ(Type::getInt8Ty(getGlobalContext()),
            (TypeBuilder<volatile types::i<8>, true>::get(getGlobalContext())));
  EXPECT_EQ(Type::getInt8Ty(getGlobalContext()),
            (TypeBuilder<const volatile types::i<8>, true>::get(getGlobalContext())));

  EXPECT_EQ(PointerType::getUnqual(Type::getInt8Ty(getGlobalContext())),
            (TypeBuilder<const volatile int8_t*const volatile, false>::get(getGlobalContext())));
}

TEST(TypeBuilderTest, Functions) {
  std::vector<const Type*> params;
  EXPECT_EQ(FunctionType::get(Type::getVoidTy(getGlobalContext()), params, false),
            (TypeBuilder<void(), true>::get(getGlobalContext())));
  EXPECT_EQ(FunctionType::get(Type::getInt8Ty(getGlobalContext()), params, true),
            (TypeBuilder<int8_t(...), false>::get(getGlobalContext())));
  params.push_back(TypeBuilder<int32_t*, false>::get(getGlobalContext()));
  EXPECT_EQ(FunctionType::get(Type::getInt8Ty(getGlobalContext()), params, false),
            (TypeBuilder<int8_t(const int32_t*), false>::get(getGlobalContext())));
  EXPECT_EQ(FunctionType::get(Type::getInt8Ty(getGlobalContext()), params, true),
            (TypeBuilder<int8_t(const int32_t*, ...), false>::get(getGlobalContext())));
  params.push_back(TypeBuilder<char*, false>::get(getGlobalContext()));
  EXPECT_EQ(FunctionType::get(Type::getInt8Ty(getGlobalContext()), params, false),
            (TypeBuilder<int8_t(int32_t*, void*), false>::get(getGlobalContext())));
  EXPECT_EQ(FunctionType::get(Type::getInt8Ty(getGlobalContext()), params, true),
            (TypeBuilder<int8_t(int32_t*, char*, ...), false>::get(getGlobalContext())));
  params.push_back(TypeBuilder<char, false>::get(getGlobalContext()));
  EXPECT_EQ(FunctionType::get(Type::getInt8Ty(getGlobalContext()), params, false),
            (TypeBuilder<int8_t(int32_t*, void*, char), false>::get(getGlobalContext())));
  EXPECT_EQ(FunctionType::get(Type::getInt8Ty(getGlobalContext()), params, true),
            (TypeBuilder<int8_t(int32_t*, char*, char, ...), false>::get(getGlobalContext())));
  params.push_back(TypeBuilder<char, false>::get(getGlobalContext()));
  EXPECT_EQ(FunctionType::get(Type::getInt8Ty(getGlobalContext()), params, false),
            (TypeBuilder<int8_t(int32_t*, void*, char, char), false>::get(getGlobalContext())));
  EXPECT_EQ(FunctionType::get(Type::getInt8Ty(getGlobalContext()), params, true),
            (TypeBuilder<int8_t(int32_t*, char*, char, char, ...),
                         false>::get(getGlobalContext())));
  params.push_back(TypeBuilder<char, false>::get(getGlobalContext()));
  EXPECT_EQ(FunctionType::get(Type::getInt8Ty(getGlobalContext()), params, false),
            (TypeBuilder<int8_t(int32_t*, void*, char, char, char),
                         false>::get(getGlobalContext())));
  EXPECT_EQ(FunctionType::get(Type::getInt8Ty(getGlobalContext()), params, true),
            (TypeBuilder<int8_t(int32_t*, char*, char, char, char, ...),
                         false>::get(getGlobalContext())));
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

class MyType {
  int a;
  int *b;
  void *array[1];
};

class MyPortableType {
  int32_t a;
  int32_t *b;
  void *array[1];
};

}  // anonymous namespace

namespace llvm {
template<bool cross> class TypeBuilder<MyType, cross> {
public:
  static const StructType *get(LLVMContext &Context) {
    // Using the static result variable ensures that the type is
    // only looked up once.
    std::vector<const Type*> st;
    st.push_back(TypeBuilder<int, cross>::get(Context));
    st.push_back(TypeBuilder<int*, cross>::get(Context));
    st.push_back(TypeBuilder<void*[], cross>::get(Context));
    static const StructType *const result = StructType::get(Context, st);
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
  static const StructType *get(LLVMContext &Context) {
    // Using the static result variable ensures that the type is
    // only looked up once.
    std::vector<const Type*> st;
    st.push_back(TypeBuilder<types::i<32>, cross>::get(Context));
    st.push_back(TypeBuilder<types::i<32>*, cross>::get(Context));
    st.push_back(TypeBuilder<types::i<8>*[], cross>::get(Context));
    static const StructType *const result = StructType::get(Context, st);
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
  EXPECT_EQ(PointerType::getUnqual(StructType::get(getGlobalContext(), 
                                     TypeBuilder<int, false>::get(getGlobalContext()),
                                     TypeBuilder<int*, false>::get(getGlobalContext()),
                                     TypeBuilder<void*[], false>::get(getGlobalContext()),
                                     NULL)),
            (TypeBuilder<MyType*, false>::get(getGlobalContext())));
  EXPECT_EQ(PointerType::getUnqual(StructType::get(getGlobalContext(), 
                                     TypeBuilder<types::i<32>, false>::get(getGlobalContext()),
                                     TypeBuilder<types::i<32>*, false>::get(getGlobalContext()),
                                     TypeBuilder<types::i<8>*[], false>::get(getGlobalContext()),
                                     NULL)),
            (TypeBuilder<MyPortableType*, false>::get(getGlobalContext())));
  EXPECT_EQ(PointerType::getUnqual(StructType::get(getGlobalContext(), 
                                     TypeBuilder<types::i<32>, false>::get(getGlobalContext()),
                                     TypeBuilder<types::i<32>*, false>::get(getGlobalContext()),
                                     TypeBuilder<types::i<8>*[], false>::get(getGlobalContext()),
                                     NULL)),
            (TypeBuilder<MyPortableType*, true>::get(getGlobalContext())));
}

}  // anonymous namespace
