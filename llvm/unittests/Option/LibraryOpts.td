//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "llvm/Option/LibraryOptions.td"

def TestLibraryOptions : OptionsStruct<"test", prefix = "lib-">;

defm enable : BoolField<"lib-enable", "false", "A bool">;
// Aliases keep an old spelling working. An alias without HelpText is hidden.
def : FlagOrEq<["-", "--"], "lib-old-enable">, Alias<enable>;
// A Flag alias supplies the target's value in AliasArgs: here, a polarity flip.
def : Flag<["-", "--"], "lib-disable">, Alias<enable>, AliasArgs<["false"]>,
  HelpText<"Same as -lib-enable=false">;
def InBuild : BuildCondition<"__cplusplus">;
def NotInBuild : BuildCondition<"__cplusplus < 201103L">;
let Condition = InBuild in
defm in_build : BoolField<"lib-in-build", "false", "Available">;
// An alias is built where its target is.
def : FlagOrEq<["-", "--"], "lib-old-in-build">, Alias<in_build>;
let Condition = NotInBuild in
defm not_in_build : BoolField<"lib-not-in-build", "false", "Unavailable">;
def : FlagOrEq<["-", "--"], "lib-old-not-in-build">, Alias<not_in_build>;
def : Flag<["-", "--"], "lib-no-not-in-build">, Alias<not_in_build>,
  AliasArgs<["false"]>;
defm count : ValueField<"lib-count", "unsigned", "3", "An unsigned">;
def : SeparateOrEq<["-", "--"], "lib-old-count">, Alias<count>;
def : Flag<["-", "--"], "lib-count-one">, Alias<count>, AliasArgs<["1"]>;
defm : OptionalValueField<"lib-limit", "unsigned", "An optional">;
defm : ListField<"lib-list", "unsigned", "A list">;
defm mode : EnumField<"lib-mode", "test::Mode", "test::Mode::A", "a,b",
  ["test::Mode::A", "test::Mode::B"], "An enum">;
def : Flag<["-"], "lib-mode-b">, Alias<mode>, AliasArgs<["b"]>;
defm : EnumListField<"lib-modes", "test::Mode", "a,b",
  ["test::Mode::A", "test::Mode::B"], "An enum list">;
defm override : OptionalBoolField<"lib-override", "An optional bool">;
def : Flag<["-"], "lib-no-override">, Alias<override>, AliasArgs<["false"]>;
defm print : FlagOrEnumField<"lib-print", "std::optional<test::Mode>",
  "std::nullopt", "a,b", ["test::Mode::A", "test::Mode::B"], "test::Mode::B",
  "A flag or enum">;
def : FlagOrEq<["-"], "lib-old-print">, Alias<print>;
defm : ValueField<"lib-ratio", "double", "0.5", "A double">;
defm : DefaultOnOffField<"lib-tristate", "A tri-state">;
defm Path : ValueField<"lib-path", "StringRef", "\"p\"", "A string">;
