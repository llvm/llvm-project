//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "llvm/Option/LibraryOptions.td"

def ARMMCOptions : OptionsStruct<prefix = "arm-">;

// AsmParser/ARMAsmParser.cpp
defm : BoolField<"arm-add-build-attributes", "false", "">;
defm : EnumField<"arm-implicit-it", "ImplicitItModeTy", "ImplicitItModeTy::ARMOnly",
  "always,never,arm,thumb", ["ImplicitItModeTy::Always", "ImplicitItModeTy::Never", "ImplicitItModeTy::ARMOnly", "ImplicitItModeTy::ThumbOnly"],
  "Allow conditional instructions outside of an IT block">;
