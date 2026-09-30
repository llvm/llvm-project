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
// `defm Name : ...` names it. Only -help-hidden lists the options, except those
// declared in `let Hidden = 0 in { ... }`.
//
//===----------------------------------------------------------------------===//

include "llvm/Option/OptParser.td"

class OptionsStruct<string namespace = "llvm"> { string Namespace = namespace; }

// Every row of a BoolField or ValueField. Hidden adds HelpHidden.
class FieldRow { bit Hidden = 1; }

// The member a row sets, the member's type and default value, and the value a
// Flag assigns. Other kinds parse their argument.
class Field<string name, string type, code default> : FieldRow {
  string FieldName = name;
  string FieldType = type;
  code FieldDefault = default;
}
class FieldValue<code value> { code FieldValue = value; }

// A bool member set by -<name> or -<name>=<true|false|1|0>.
multiclass BoolField<string name, code default, string help> {
  def NAME : Flag<["-", "--"], name>, Field<NAME, "bool", default>,
    FieldValue<"true">, HelpText<help> {
    let Flags = !if(Hidden, [HelpHidden], []);
  }
  def NAME#_EQ : Joined<["-", "--"], name#"=">, Field<NAME, "bool", default>;
}

// A member of the C++ type <type> set by -<name>=<value> or -<name> <value>.
multiclass ValueField<string name, string type, code default, string help> {
  def NAME#_EQ : Joined<["-", "--"], name#"=">, Field<NAME, type, default>,
    HelpText<help> {
    let Flags = !if(Hidden, [HelpHidden], []);
  }
  def NAME : Separate<["-", "--"], name>, Alias<!cast<Option>(NAME#"_EQ")>,
    FieldRow;
}
