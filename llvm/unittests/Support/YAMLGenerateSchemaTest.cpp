//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/YAMLGenerateSchema.h"
#include "llvm/Support/YAMLTraits.h"
#include <optional>
#include <string>
#include <vector>

#include "gtest/gtest.h"

using namespace llvm;

enum class ColorTy {
  White,
  Black,
  Blue,
};

struct Baby {
  std::string Name;
  ColorTy Color;
};

LLVM_YAML_IS_SEQUENCE_VECTOR(Baby)

struct Animal {
  std::string Name;
  std::optional<int> Age;
  std::vector<Baby> Babies;
};

LLVM_YAML_IS_SEQUENCE_VECTOR(Animal)

namespace llvm {
namespace yaml {

template <> struct ScalarEnumerationTraits<ColorTy> {
  static void enumeration(IO &io, ColorTy &value) {
    io.enumCase(value, "white", ColorTy::White);
    io.enumCase(value, "black", ColorTy::Black);
    io.enumCase(value, "blue", ColorTy::Blue);
  }
};

template <> struct MappingTraits<Baby> {
  static void mapping(IO &io, Baby &info) {
    io.mapRequired("name", info.Name);
    io.mapRequired("color", info.Color);
  }
};

template <> struct MappingTraits<Animal> {
  static void mapping(IO &io, Animal &info) {
    io.mapRequired("name", info.Name);
    io.mapOptional("age", info.Age);
    io.mapOptional("babies", info.Babies);
  }
};

} // namespace yaml
} // namespace llvm

TEST(ObjectYAMLGenerateSchema, SimpleSchema) {
  std::string String;
  raw_string_ostream OS(String);
  std::vector<Animal> Animals;
  yaml::GenerateSchema Gen(OS);
  Gen << Animals;
  StringRef YAMLSchema = R"({
  "flowStyle": "block",
  "items": {
    "flowStyle": "block",
    "optional": [
      "age",
      "babies"
    ],
    "properties": {
      "age": {
        "type": "string"
      },
      "babies": {
        "flowStyle": "block",
        "items": {
          "flowStyle": "block",
          "properties": {
            "color": {
              "enum": [
                "white",
                "black",
                "blue"
              ],
              "type": "string"
            },
            "name": {
              "type": "string"
            }
          },
          "required": [
            "name",
            "color"
          ],
          "type": "object"
        },
        "type": "array"
      },
      "name": {
        "type": "string"
      }
    },
    "required": [
      "name"
    ],
    "type": "object"
  },
  "type": "array"
}
)";
  EXPECT_EQ(String.c_str(), YAMLSchema.str());
}
