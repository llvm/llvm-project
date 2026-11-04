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
  // A field of `?` is `?`.
  string yStr = y.str;
  bit yStrIsSet = !initialized(y.str);
  // A class is instantiated with `?` for an unset field.
  bit strIsSet = IsSet<str>.ret;
  // Dumps see `?`.
  dump "str = " # !repr(str);
}

// CHECK-LABEL: def FieldsSet {
// CHECK:   string str = "s";
// CHECK:   Y y = Y1;
// CHECK:   string yStr = "y1";
// CHECK:   bit yStrIsSet = 1;
// CHECK:   bit strIsSet = 1;
// CHECK-LABEL: def FieldsUnset {
// CHECK:   string str = ?;
// CHECK:   Y y = ?;
// CHECK:   string yStr = ?;
// CHECK:   bit yStrIsSet = 0;
// CHECK:   bit strIsSet = 0;
// DUMP: note: str = ?
// DUMP: note: str = "s"
def FieldsUnset : Fields;
def FieldsSet : Fields {
  let str = "s";
  let y = Y1;
}

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

#ifdef ERROR
// Assertions see `?`.
class Asserted {
  Y y = ?;
  assert !initialized(y.str), "y.str is unset";
}
// ERROR: error: assertion failed: y.str is unset
def AssertedUnset : Asserted;
#endif
