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

// The member a row sets, the member's type and default value, and the value a
// Flag assigns. Other kinds parse their argument.
class Field<string name, string type, code default> {
  string FieldName = name;
  string FieldType = type;
  code FieldDefault = default;
}
class FieldValue<code value> { code FieldValue = value; }

// A bool member set by -<name> or -<name>=<true|false|1|0>.
multiclass BoolField<string name, code default, string help> {
  def NAME : Flag<["-", "--"], name>, Field<NAME, "bool", default>,
    FieldValue<"true">, HelpText<help>;
  def NAME#_EQ : Joined<["-", "--"], name#"=">, Field<NAME, "bool", default>;
}

// A std::optional<bool> member, std::nullopt unless -<name> or
// -<name>=<true|false|1|0> is given.
multiclass OptionalBoolField<string name, string help> {
  def NAME : Flag<["-", "--"], name>,
    Field<NAME, "std::optional<bool>", "std::nullopt">, FieldValue<"true">,
    HelpText<help>;
  def NAME#_EQ : Joined<["-", "--"], name#"=">,
    Field<NAME, "std::optional<bool>", "std::nullopt">;
}

// A member of the enumeration <type> set by -<name>=<value> or -<name> <value>,
// where <value> is one of the comma-separated <values>, which map in order to
// the C++ <enumerators>.
multiclass EnumField<string name, string type, code default, string values,
                     list<code> enumerators, string help> {
  def NAME#_EQ : Joined<["-", "--"], name#"=">, Field<NAME, type, default>,
    Values<values>, NormalizedValues<enumerators>,
    MetaVarName<"<"#!subst(",", "|", values)#">">, HelpText<help>;
  def NAME : Separate<["-", "--"], name>, Alias<!cast<Option>(NAME#"_EQ")>;
}

// A std::optional<bool> member, std::nullopt unless -<name>=Enable or
// -<name>=Disable is given.
multiclass DefaultOnOffField<string name, string help> {
  defm NAME : EnumField<name, "std::optional<bool>", "std::nullopt",
    "Default,Enable,Disable", ["std::nullopt", "true", "false"], help>;
}

// A member of the C++ type <type> set by -<name>=<value> or -<name> <value>.
multiclass ValueField<string name, string type, code default, string help> {
  def NAME#_EQ : Joined<["-", "--"], name#"=">, Field<NAME, type, default>,
    HelpText<help>;
  def NAME : Separate<["-", "--"], name>, Alias<!cast<Option>(NAME#"_EQ")>;
}
