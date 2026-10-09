//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Declaring a library's options as the members of a struct. A .td with an
// OptionsStruct def generates a struct named after the def in \p namespace,
// with one member per BoolField or ValueField.
// `defm : BoolField<"foo-bar", ...>` declares member foo_bar, and
// `defm Name : ...` names it. Only -help-hidden lists the options.
//
//===----------------------------------------------------------------------===//

include "llvm/Option/OptParser.td"

// A row whose spelling starts with prefix names its member without it.
class OptionsStruct<string namespace = "llvm", string prefix = ""> {
  string Namespace = namespace;
  string MemberPrefix = prefix;
}

// The member an option sets, and the member's type and default value.
class Field<string name, string type, code default> {
  string FieldName = name;
  string FieldType = type;
  code FieldDefault = default;
}

// The value a FlagOrEq row assigns when given without one.
class BareValue<code value> { code BareValue = value; }

// The comma-separated <values> of an enum row, which map in order to the C++
// <enumerators>.
class EnumValues<string values, list<code> enumerators>
    : Values<values>, NormalizedValues<enumerators>,
      MetaVarName<"<"#!subst(",", "|", values)#">">;

// A bool member set by -<name> or -<name>=<true|false|1|0>.
multiclass BoolField<string name, code default, string help> {
  def NAME : FlagOrEq<["-", "--"], name>, Field<NAME, "bool", default>,
    BareValue<"true">, HelpText<help>;
}

// A BoolOrDefault member, Default unless -<name> or -<name>=<true|false|1|0>
// is given.
multiclass OptionalBoolField<string name, string help> {
  def NAME : FlagOrEq<["-", "--"], name>,
    Field<NAME, "llvm::BoolOrDefault", "llvm::BoolOrDefault::Default">,
    BareValue<"llvm::BoolOrDefault::True">, HelpText<help>;
}

// A member of the enumeration <type> set by -<name>=<value> or -<name> <value>,
// where <value> is one of the comma-separated <values>, which map in order to
// the C++ <enumerators>.
multiclass EnumField<string name, string type, code default, string values,
                     list<code> enumerators, string help> {
  def NAME : SeparateOrEq<["-", "--"], name>, Field<NAME, type, default>,
    EnumValues<values, enumerators>, HelpText<help>;
}

// An EnumField also set to <bare> by -<name> without a value. It never consumes
// the next argument.
multiclass FlagOrEnumField<string name, string type, code default,
                           string values, list<code> enumerators, code bare,
                           string help> {
  def NAME : FlagOrEq<["-", "--"], name>, Field<NAME, type, default>,
    EnumValues<values, enumerators>, BareValue<bare>, HelpText<help>;
}

// A BoolOrDefault member, Default unless -<name>=Enable or -<name>=Disable is
// given.
multiclass DefaultOnOffField<string name, string help> {
  defm NAME : EnumField<name, "llvm::BoolOrDefault",
    "llvm::BoolOrDefault::Default", "Default,Enable,Disable",
    ["llvm::BoolOrDefault::Default", "llvm::BoolOrDefault::True",
     "llvm::BoolOrDefault::False"], help>;
}

// A member of the C++ type <type> set by -<name>=<value> or -<name> <value>.
multiclass ValueField<string name, string type, code default, string help> {
  def NAME : SeparateOrEq<["-", "--"], name>, Field<NAME, type, default>,
    HelpText<help>;
}
