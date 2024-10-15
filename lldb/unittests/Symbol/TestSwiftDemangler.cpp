#include "gtest/gtest.h"

#include "Plugins/LanguageRuntime/Swift/SwiftLanguageRuntime.h"

using namespace lldb;
using namespace lldb_private;
using namespace llvm;
static constexpr auto IsSwiftMangledName =
    SwiftLanguageRuntime::IsSwiftMangledName;
static constexpr auto IsAnySwiftAsyncFunctionSymbol = [](StringRef name) {
  return SwiftLanguageRuntime::IsAnySwiftAsyncFunctionSymbol(name);
};

TEST(TestSwiftDemangleAsyncNames, BasicAsync) {
  // "sayBasic" == a basic async function
  // "sayGeneric" == a generic async function
  SmallVector<StringRef> basic_funclets = {
      "$s1a8sayBasicyySSYaF",
      "$s1a8sayBasicyySSYaFTY0_",
      "$s1a8sayBasicyySSYaFTQ1_",
      "$s1a8sayBasicyySSYaFTY2_",
  };
  SmallVector<StringRef> generic_funclets = {
      "$s1a10sayGenericyyxYalF",
      "$s1a10sayGenericyyxYalFTY0_",
      "$s1a10sayGenericyyxYalFTQ1_",
      "$s1a10sayGenericyyxYalFTY2_",
  };
  for (StringRef async_name :
       llvm::concat<StringRef>(basic_funclets, generic_funclets)) {
    EXPECT_TRUE(IsSwiftMangledName(async_name)) << async_name;
    EXPECT_TRUE(IsAnySwiftAsyncFunctionSymbol(async_name)) << async_name;
  }
}

TEST(TestSwiftDemangleAsyncNames, ClosureAsync) {
  // These are all async closures
  SmallVector<StringRef> nested1_funclets = {
      // Nesting level 1: a closure inside a function.
      "$s1a8sayHelloyyYaFyypYacfU_",     "$s1a8sayHelloyyYaFyypYacfU_TY0_",
      "$s1a8sayHelloyyYaFyypYacfU_TQ1_", "$s1a8sayHelloyyYaFyypYacfU_TY2_",
      "$s1a8sayHelloyyYaFyypYacfU_TQ3_", "$s1a8sayHelloyyYaFyypYacfU_TY4_",
      "$s1a8sayHelloyyYaFyypYacfU_TQ5_", "$s1a8sayHelloyyYaFyypYacfU_TY6_"};
  SmallVector<StringRef> nested2_funclets1 = {
      // Nesting level 2: a closure inside a closure.
      "$s1a8sayHelloyyYaFyypYacfU_yypYacfU_",
      "$s1a8sayHelloyyYaFyypYacfU_yypYacfU_TY0_",
      "$s1a8sayHelloyyYaFyypYacfU_yypYacfU_TQ1_",
      "$s1a8sayHelloyyYaFyypYacfU_yypYacfU_TY2_",
  };
  SmallVector<StringRef> nested2_funclets2 = {
      // Nesting level 2: another closure, same level as the previous one.
      "$s1a8sayHelloyyYaFyypYacfU_yypYacfU0_",
      "$s1a8sayHelloyyYaFyypYacfU_yypYacfU0_TY0_",
      "$s1a8sayHelloyyYaFyypYacfU_yypYacfU0_TQ1_",
      "$s1a8sayHelloyyYaFyypYacfU_yypYacfU0_TY2_",
  };
  SmallVector<StringRef> nested2_funclets_top_not_async = {
      // Also nesting level 2: but this time, the top level function is _not_
      // async!
      "$s1a18myNonAsyncFunctionyyFyyYacfU_SiypYacfU_SSypYacfU0_",
      "$s1a18myNonAsyncFunctionyyFyyYacfU_SiypYacfU_SSypYacfU0_TY0_",
      "$s1a18myNonAsyncFunctionyyFyyYacfU_SiypYacfU_SSypYacfU0_TQ1_",
      "$s1a18myNonAsyncFunctionyyFyyYacfU_SiypYacfU_SSypYacfU0_TY2_"};

  for (StringRef async_name : llvm::concat<StringRef>(
           nested1_funclets, nested2_funclets1, nested2_funclets2,
           nested2_funclets_top_not_async)) {
    EXPECT_TRUE(IsSwiftMangledName(async_name)) << async_name;
    EXPECT_TRUE(IsAnySwiftAsyncFunctionSymbol(async_name)) << async_name;
  }
}

TEST(TestSwiftDemangleAsyncNames, StaticAsync) {
  // static async functions
  SmallVector<StringRef> async_names = {
      "$s1a6StructV9sayStaticyySSYaFZ"
      "$s1a6StructV9sayStaticyySSYaFZTY0_",
      "$s1a6StructV9sayStaticyySSYaFZTQ1_",
      "$s1a6StructV9sayStaticyySSYaFZTY2_",
  };

  for (StringRef async_name : async_names) {
    EXPECT_TRUE(IsSwiftMangledName(async_name)) << async_name;
    EXPECT_TRUE(IsAnySwiftAsyncFunctionSymbol(async_name)) << async_name;
  }
}
