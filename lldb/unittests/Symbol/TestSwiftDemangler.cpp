#include "gtest/gtest.h"

#include "Plugins/LanguageRuntime/Swift/SwiftLanguageRuntime.h"
#include "llvm/ADT/Sequence.h"

using namespace lldb;
using namespace lldb_private;
using namespace llvm;
static constexpr auto IsSwiftMangledName =
    SwiftLanguageRuntime::IsSwiftMangledName;
static constexpr auto GetFuncletNumber = [](StringRef name) {
  return SwiftLanguageRuntime::GetFuncletNumber(name);
};
static constexpr auto IsAnySwiftAsyncFunctionSymbol = [](StringRef name) {
  return SwiftLanguageRuntime::IsAnySwiftAsyncFunctionSymbol(name);
};
static constexpr auto AreFuncletsOfSameAsyncFunction = [](StringRef name1,
                                                          StringRef name2) {
  return SwiftLanguageRuntime::AreFuncletsOfSameAsyncFunction(name1, name2);
};


using FuncletComparisonResult = SwiftLanguageRuntime::FuncletComparisonResult;

/// Checks that all names in \c funclets belong to the same function.
static void CheckGroupOfFuncletsFromSameFunction(ArrayRef<StringRef> funclets) {
  for (StringRef funclet1 : funclets)
    for (StringRef funclet2 : funclets) {
      EXPECT_EQ(FuncletComparisonResult::SameAsyncFunction,
                AreFuncletsOfSameAsyncFunction(funclet1, funclet2))
          << funclet1 << " -- " << funclet2;
      EXPECT_EQ(FuncletComparisonResult::SameAsyncFunction,
                AreFuncletsOfSameAsyncFunction(funclet2, funclet1))
          << funclet1 << " -- " << funclet2;
    }
}

/// Checks that all pairs of combinations of names from \c funclets1 and \c
/// funclets2 belong to different functions.
static void
CheckGroupOfFuncletsFromDifferentFunctions(ArrayRef<StringRef> funclets1,
                                           ArrayRef<StringRef> funclets2) {
  for (StringRef funclet1 : funclets1)
    for (StringRef funclet2 : funclets2) {
      EXPECT_EQ(FuncletComparisonResult::DifferentAsyncFunctions,
                AreFuncletsOfSameAsyncFunction(funclet1, funclet2))
          << funclet1 << " -- " << funclet2;
      EXPECT_EQ(FuncletComparisonResult::DifferentAsyncFunctions,
                AreFuncletsOfSameAsyncFunction(funclet2, funclet1))
          << funclet1 << " -- " << funclet2;
    }
}

/// Check that funclets contain a sequence of funclet names whose "async
/// numbers" go from 0 to size(funclets).
static void CheckFuncletNumbersAreARange(ArrayRef<StringRef> funclets) {
  for (auto idx : llvm::seq<int>(0, funclets.size()))
    EXPECT_EQ(idx, GetFuncletNumber(funclets[idx]));
}

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

  CheckGroupOfFuncletsFromSameFunction(basic_funclets);
  CheckGroupOfFuncletsFromSameFunction(generic_funclets);
  CheckGroupOfFuncletsFromDifferentFunctions(basic_funclets, generic_funclets);
  CheckFuncletNumbersAreARange(basic_funclets);
  CheckFuncletNumbersAreARange(generic_funclets);
}

// The funclets below are created from this program:
// swiftc -g -Onone test.swift -o - -emit-ir -module-name a \
// | grep "define.*sayHello"
// func work() async {}
// func async_int() async -> Int { return 42; }
// func sayHello() async {
//   let closure: (Any) async -> () = { _ in
//     print("hello")
//     await work()
//     print("hello")
//
//     let inner_closure: (Any) async -> () = { _ in
//       print("hello")
//       await work()
//       print("hello")
//     }
//     await inner_closure(10)
//     print("hello")
//
//     let inner_closure2: (Any) async -> () = { _ in
//       print("hello")
//       await work()
//       print("hello")
//     }
//
//     await inner_closure2(10)
//     print("hello")
//     async let x = await async_int();
//     print(await x);
//   }
//   async let x = await async_int();
//   print(await x);
//   await closure(10)
//   async let explicit_inside_implicit_closure =
//     { _ in
//       print("hello")
//       await work()
//       print("hello")
//       return 42
//     }(10)
//   print(await explicit_inside_implicit_closure)
// }
// func sayHello2() async {
//  {_ in
//  }(10)
//  async let another_explicit_inside_implicit_closure =
//    { _ in
//      print("hello")
//      await work()
//      print("hello")
//      return 42
//    }(10)
//  print(await another_explicit_inside_implicit_closure)
//}

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

  SmallVector<StringRef> implicit_closure_inside_function = {
      "$s1a8sayHelloyyYaFSiyYaYbcfu_",
      "$s1a8sayHelloyyYaFSiyYaYbcfu_TQ0_",
      "$s1a8sayHelloyyYaFSiyYaYbcfu_TY1_",
  };
  SmallVector<StringRef> implicit_closure_inside_explicit_closure = {
      "$s1a8sayHelloyyYaFyypYacfU_SiyYaYbcfu_",
      "$s1a8sayHelloyyYaFyypYacfU_SiyYaYbcfu_TQ0_",
      "$s1a8sayHelloyyYaFyypYacfU_SiyYaYbcfu_TY1_",
  };
  SmallVector<StringRef> explicit_closure_inside_implicit_closure = {
      "$s1a8sayHelloyyYaFSiyYaYbcfu0_S2iYaXEfU0_",
      "$s1a8sayHelloyyYaFSiyYaYbcfu0_S2iYaXEfU0_TY0_",
      "$s1a8sayHelloyyYaFSiyYaYbcfu0_S2iYaXEfU0_TQ1_",
      "$s1a8sayHelloyyYaFSiyYaYbcfu0_S2iYaXEfU0_TY2_",
  };
  SmallVector<StringRef> another_explicit_closure_inside_implicit_closure = {
      "$s1a9sayHello2yyYaFSiyYaYbcfu_S2iYaXEfU0_",
      "$s1a9sayHello2yyYaFSiyYaYbcfu_S2iYaXEfU0_TY0_",
      "$s1a9sayHello2yyYaFSiyYaYbcfu_S2iYaXEfU0_TQ1_",
      "$s1a9sayHello2yyYaFSiyYaYbcfu_S2iYaXEfU0_TY2_",
  };

  SmallVector<ArrayRef<StringRef>, 0> funclet_groups = {
      nested1_funclets,
      nested2_funclets1,
      nested2_funclets2,
      nested2_funclets_top_not_async,
      implicit_closure_inside_function,
      implicit_closure_inside_explicit_closure,
      explicit_closure_inside_implicit_closure,
      another_explicit_closure_inside_implicit_closure,
  };

  for (ArrayRef<StringRef> funclet_group : funclet_groups)
    for (StringRef async_name : funclet_group) {
      EXPECT_TRUE(IsSwiftMangledName(async_name)) << async_name;
      EXPECT_TRUE(IsAnySwiftAsyncFunctionSymbol(async_name)) << async_name;
    }

  for (ArrayRef<StringRef> funclet_group : funclet_groups)
    CheckGroupOfFuncletsFromSameFunction(funclet_group);

  for (ArrayRef<StringRef> funclet_group1 : funclet_groups)
    for (ArrayRef<StringRef> funclet_group2 : funclet_groups)
      if (funclet_group1.data() != funclet_group2.data())
        CheckGroupOfFuncletsFromDifferentFunctions(funclet_group1,
                                                   funclet_group2);

  for (ArrayRef<StringRef> funclet_group : funclet_groups)
    CheckFuncletNumbersAreARange(funclet_group);
}

TEST(TestSwiftDemangleAsyncNames, StaticAsync) {
  // static async functions
  SmallVector<StringRef> static_async_funclets = {
      "$s1a6StructV9sayStaticyySSYaFZ",
      "$s1a6StructV9sayStaticyySSYaFZTY0_",
      "$s1a6StructV9sayStaticyySSYaFZTQ1_",
      "$s1a6StructV9sayStaticyySSYaFZTY2_",
  };

  for (StringRef async_name : static_async_funclets) {
    EXPECT_TRUE(IsSwiftMangledName(async_name)) << async_name;
    EXPECT_TRUE(IsAnySwiftAsyncFunctionSymbol(async_name)) << async_name;
  }

  CheckGroupOfFuncletsFromSameFunction(static_async_funclets);

  // Make sure we can compare static funclets to other kinds of funclets
  SmallVector<StringRef> other_funclets = {
      // Nested funclets:
      "$s1a8sayHelloyyYaFyypYacfU_", "$s1a8sayHelloyyYaFyypYacfU_TY0_",
      // "Normal" funclets:
      "$s1a8sayBasicyySSYaF", "$s1a8sayBasicyySSYaFTY0_"};
  CheckGroupOfFuncletsFromDifferentFunctions(static_async_funclets,
                                             other_funclets);
  CheckFuncletNumbersAreARange(static_async_funclets);
}

TEST(TestSwiftDemangleAsyncNames, NonAsync) {
  // func factorial(_ n:Int) -> Int {
  {
    StringRef func = "$s4test9factorialyS2iF";
    EXPECT_EQ(GetFuncletNumber(func), std::nullopt);
  }

  // func factorial(_ n:Int) async -> Int {
  //   func inner_factorial(_ n:Int) -> Int {
  {
    StringRef func = "$s4test9factorialyS2iYaF06inner_B0L_yS2iF";
    EXPECT_EQ(GetFuncletNumber(func), std::nullopt);
  }
}
