//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "llvm/Option/LibraryOptions.td"

def TestLibraryOptions : OptionsStruct<"test", prefix = "lib-">;

defm : BoolField<"lib-enable", "false", "A bool">;
defm : ValueField<"lib-count", "unsigned", "3", "An unsigned">;
defm : OptionalValueField<"lib-limit", "unsigned", "An optional">;
defm : ListField<"lib-list", "unsigned", "A list">;
defm : EnumField<"lib-mode", "test::Mode", "test::Mode::A", "a,b",
  ["test::Mode::A", "test::Mode::B"], "An enum">;
defm : EnumListField<"lib-modes", "test::Mode", "a,b",
  ["test::Mode::A", "test::Mode::B"], "An enum list">;
defm : OptionalBoolField<"lib-override", "An optional bool">;
defm : FlagOrEnumField<"lib-print", "std::optional<test::Mode>", "std::nullopt",
  "a,b", ["test::Mode::A", "test::Mode::B"], "test::Mode::B",
  "A flag or enum">;
defm : ValueField<"lib-ratio", "double", "0.5", "A double">;
defm : DefaultOnOffField<"lib-tristate", "A tri-state">;
defm Path : ValueField<"lib-path", "StringRef", "\"p\"", "A string">;
