// RUN: llvm-tblgen %s | FileCheck %s
// RUN: llvm-tblgen %s 2>&1 >/dev/null | FileCheck %s --check-prefix=DUMP
// RUN: not llvm-tblgen -DERROR %s 2>&1 | FileCheck %s --check-prefix=ERROR

// Once the record is complete, a reference to one of its unset fields is `?`,
// like a template argument that is `?` or an unset field of another record.
// (References to unset bits are kept, see JSON.td.)

class Y {
  string str = ?;
}
def Y1 : Y {
  let str = "y1";
}

class IsSet<string s> {
  bit ret = !initialized(s);
}

class Fields {
  string str = ?;
  Y y = ?;
  // `!initialized` folds.
  bit strIsInit = !initialized(str);
  // A field of `?` is `?`.
  string yStr = y.str;
  bit yStrIsSet = !initialized(y.str);
  // A class is instantiated with `?` for an unset field.
  bit strIsSet = IsSet<str>.ret;
  // Dumps and assertions see `?`.
  dump "str = " # !repr(str);
  assert !if(!initialized(str), !ne(str, ""), 1), "str must not be empty";
}
class FieldsWithStr : Fields {
  let str = "sub";
}

// CHECK-LABEL: def FieldsLetIn {
// CHECK:   string str = "let-in";
// CHECK:   bit strIsInit = 1;
// CHECK:   bit strIsSet = 1;
// CHECK-LABEL: def FieldsSet {
// CHECK:   string str = "s";
// CHECK:   Y y = Y1;
// CHECK:   bit strIsInit = 1;
// CHECK:   string yStr = "y1";
// CHECK:   bit yStrIsSet = 1;
// CHECK:   bit strIsSet = 1;
// CHECK-LABEL: def FieldsSub {
// CHECK:   string str = "sub";
// CHECK:   bit strIsInit = 1;
// CHECK:   bit strIsSet = 1;
// CHECK-LABEL: def FieldsUnset {
// CHECK:   string str = ?;
// CHECK:   Y y = ?;
// CHECK:   bit strIsInit = 0;
// CHECK:   string yStr = ?;
// CHECK:   bit yStrIsSet = 0;
// CHECK:   bit strIsSet = 0;
// DUMP: note: str = ?
// DUMP: note: str = "s"
// DUMP: note: str = "let-in"
// DUMP: note: str = "sub"
def FieldsUnset : Fields;
def FieldsSet : Fields {
  let str = "s";
  let y = Y1;
}
let str = "let-in" in
def FieldsLetIn : Fields;
def FieldsSub : FieldsWithStr;

// A field of a template argument that is `?` is `?` as well.
class FromArg<Y y> {
  string yStr = y.str;
}
// CHECK-LABEL: def FromSetArg {
// CHECK:   string yStr = "y1";
// CHECK-LABEL: def FromUnsetArg {
// CHECK:   string yStr = ?;
def FromUnsetArg : FromArg<?>;
def FromSetArg : FromArg<Y1>;

// Records defined by a multiclass are complete once instantiated by `defm`.
multiclass MC<string s> {
  def _Unset : Fields;
  def _Set : Fields {
    let str = s;
  }
}
// CHECK-LABEL: def MC_Set {
// CHECK:   string str = "m";
// CHECK:   bit strIsInit = 1;
// CHECK:   bit strIsSet = 1;
// CHECK-LABEL: def MC_Unset {
// CHECK:   string str = ?;
// CHECK:   bit strIsInit = 0;
// CHECK:   bit strIsSet = 0;
defm MC : MC<"m">;

// The anonymous record created by a class invocation is complete as well: its
// unset fields are `?`, and the classes it invokes with them are instantiated
// with `?`.
class Optional<string fallback> {
  string val = ?;
  string ret = !if(!initialized(val), val, fallback);
  bit valIsSet = IsSet<val>.ret;
}
// CHECK-LABEL: def UsesOptional {
// CHECK:   string ret = "fallback";
// CHECK:   bit valIsSet = 0;
def UsesOptional {
  string ret = Optional<"fallback">.ret;
  bit valIsSet = Optional<"fallback">.valIsSet;
}

#ifdef ERROR
// Assertions see `?`.
class Asserted {
  Y y = ?;
  assert !initialized(y.str), "y.str is unset";
}
// ERROR: error: assertion failed: y.str is unset
def AssertedUnset : Asserted;

// A class instantiated with an unset field sees `?`, also in its assertions,
// even if its result is unused.
class Required<int n> {
  assert !initialized(n), "n is unset";
  int ret = n;
}
class UsesRequired<int fallback> {
  int val = ?;
  int required = Required<val>.ret;
  int ret = fallback;
}
// ERROR: error: assertion failed: n is unset
def UsesRequiredUnset {
  int ret = UsesRequired<3>.ret;
}

// The error shows the unset fields that a value depends on, not `?`.
// ERROR: error: Initializer of 'b' in 'Unresolved' could not be fully resolved: !add(a, 1)
def Unresolved {
  int a;
  int b = !add(a, 1);
}
#endif
