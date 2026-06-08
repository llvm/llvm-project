// RUN: llvm-tblgen %s | FileCheck %s
// RUN: not llvm-tblgen -DERROR_NO_DEFAULT %s 2>&1 | FileCheck --check-prefix=ERROR_NO_DEFAULT %s
// RUN: not llvm-tblgen -DERROR_NO_CASE %s 2>&1 | FileCheck --check-prefix=ERROR_NO_CASE %s
// RUN: not llvm-tblgen -DERROR_VAL_TYPE %s 2>&1 | FileCheck --check-prefix=ERROR_VAL_TYPE %s
// RUN: not llvm-tblgen -DERROR_KEY_TYPE %s 2>&1 | FileCheck --check-prefix=ERROR_KEY_TYPE %s
// RUN: not llvm-tblgen -DERROR_ALL_UNSET %s 2>&1 | FileCheck --check-prefix=ERROR_ALL_UNSET %s

class Rec<int N> { int v = N; }
def rec_a : Rec<10>;
def rec_b : Rec<20>;
def rec_c : Rec<30>;
def rec_d : Rec<40>;

// Regression test for a crash in ParseOperationSwitch - convertInitializerTo()
// dereferenced the null target type in a defvar context.
// (99 / "k99") misses, so the mandatory default is selected.
// CHECK-LABEL: def defvar_default
// CHECK:   color = "default";
// CHECK:   age = 10;
// CHECK:   rec_val = rec_a;
// CHECK:   string_key = "v1";
// CHECK:   bits_val = { 1, 1 };
// CHECK:   single_bit = 1;
// CHECK:   list_val = [7, 8, 9];
defvar dv_color = !switch(99, 1: "red", 2: "green", 3: "blue", "default");
defvar dv_age = !switch(99, 1: 20, 2: 30, 3: 40, 10);
defvar dv_rec_val = !switch(99, 1: rec_b, 2: rec_c, 3: rec_d, rec_a);
defvar dv_string_key = !switch("k99", "k1": "v2", "k2": "v3", "k3": "v4", "v1");
defvar dv_bits_val = !switch(99, 1: {0, 1}, 2: {1, 0}, {1, 1});
defvar dv_single_bit = !switch(99, 1: !eq(0, 0), 2: !eq(0, 1), !eq(1, 1));
defvar dv_list_val = !switch(99, 1: [1, 2, 3], 2: [4, 5, 6], [7, 8, 9]);
def defvar_default {
  string color = dv_color;
  int age = dv_age;
  Rec rec_val = dv_rec_val;
  string string_key = dv_string_key;
  bits<2> bits_val = dv_bits_val;
  bit single_bit = dv_single_bit;
  list<int> list_val = dv_list_val;
}

// Used inside a class (resolution is deferred until template-arg substitution)
class Labeled<int N> {
  string label = !switch(N, 1: "one", 2: "two", 3: "three", "many");
}
// CHECK-LABEL: def labeled_3
// CHECK:   label = "three";
def labeled_3 : Labeled<3>;
// CHECK-LABEL: def labeled_99
// CHECK:   label = "many";
def labeled_99 : Labeled<99>;

// Non-empty case list, no match - fallback to default
// CHECK-LABEL: def nonempty_caselist_default
// CHECK:   color  = "default";
// CHECK:   age  = 10;
// CHECK:   rec_val = rec_a;
// CHECK:   string_key = "v1";
// CHECK:   bits_val = { 1, 1 };
// CHECK:   single_bit = 1;
// CHECK:   list_val = [7, 8, 9];
def nonempty_caselist_default {
  string color = !switch(99, 1: "red", 2: "green", 3: "blue", "default");
  int age = !switch(99, 1: 20, 2: 30, 3: 40, 10);
  Rec rec_val = !switch(99, 1: rec_b, 2: rec_c, 3: rec_d, rec_a);
  string string_key = !switch("k99", "k1": "v2", "k2": "v3", "k3": "v4", "v1");
  bits<2> bits_val = !switch(99, 1: {0, 1}, 2: {1, 0}, {1, 1});
  bit single_bit = !switch(99, 1: !eq(0, 0), 2: !eq(0, 1), !eq(1, 1));
  list<int> list_val = !switch(99, 1: [1, 2, 3], 2: [4, 5, 6], [7, 8, 9]);
}

// Non-empty case list, key matches a case key
// CHECK-LABEL: def nonempty_caselist_match_case
// CHECK:   color = "red";
// CHECK:   age = 30;
// CHECK:   rec_val = rec_d;
// CHECK:   string_key = "v2";
// CHECK:   bits_val = { 1, 0 };
// CHECK:   single_bit = 0;
// CHECK:   list_val = [4, 5, 6];
def nonempty_caselist_match_case {
  string color = !switch(1,  1: "red", 2: "green", 3: "blue", "default");
  int age = !switch(2, 1: 20, 2: 30, 3: 40, 10);
  Rec rec_val = !switch(3, 1: rec_b, 2: rec_c, 3: rec_d, rec_a);
  string string_key = !switch("k1", "k1": "v2", "k2": "v3", "k3": "v4", "v1");
  bits<2> bits_val = !switch(2, 1: {0, 1}, 2: {1, 0}, {1, 1});
  bit single_bit = !switch(2, 1: !eq(0, 0), 2: !eq(0, 1), !eq(1, 1));
  list<int> list_val = !switch(2, 1: [1, 2, 3], 2: [4, 5, 6], [7, 8, 9]);
}

// Partially unset values
// CHECK-LABEL: def partially_unset_vals
// CHECK:   set_val = 5;
// CHECK:   unset_val = ?;
def partially_unset_vals {
  int set_val = !switch(2, 1: ?, 2: 5, 99);
  int unset_val  = !switch(1, 1: ?, 2: 5, 99);
}

// !switch desugars to !cond(!eq(K, ci): vi, ..., true: vd) — verify the two
// forms produce identical results.
// CHECK-LABEL: def reduction_equivalence
// CHECK:   sw = "two";
// CHECK:   eq = "two";
def reduction_equivalence {
  string sw = !switch(2, 1: "one", 2: "two", 3: "three", "other");
  string eq = !cond(!eq(2, 1): "one", !eq(2, 2): "two",
                    !eq(2, 3): "three", true: "other");
}

#ifdef ERROR_NO_DEFAULT
// Missing default: parser sees ')' where it expects another ',' after the
// last case value.
// ERROR_NO_DEFAULT: switch.td:[[@LINE+2]]:{{[0-9]+}}: error: expected ',' after case value in !switch operator
def err_no_default {
  int bad = !switch(1, 1: 10, 2: 20);
}
#endif

#ifdef ERROR_NO_CASE
// Empty case list is not allowed
// ERROR_NO_CASE: switch.td:[[@LINE+2]]:{{[0-9]+}}: error: there should be at least 1 'case: value' in the !switch operator
def err_no_case {
  int bad = !switch(1, 99);
}
#endif

#ifdef ERROR_VAL_TYPE
// Case values must be resolved to a single type (including default).
// ERROR_VAL_TYPE: switch.td:[[@LINE+2]]:{{[0-9]+}}: error: inconsistent types 'int' and 'string' for !switch values
def err_val_type {
  int bad = !switch(1, 1: 5, 2: "two", 0);
}
#endif

#ifdef ERROR_KEY_TYPE
// Key and case keys must be resolved to a single type, so the synthesized !eq calls
// type-check.
// ERROR_KEY_TYPE: switch.td:[[@LINE+2]]:{{[0-9]+}}: error: inconsistent types 'string' and 'int' between !switch key and case keys
def err_key_type {
  int bad = !switch("hi", 1: 5, 2: 6, 0);
}
#endif

#ifdef ERROR_ALL_UNSET
// All case values and default unset: value type cannot be determined.
// ERROR_ALL_UNSET: switch.td:[[@LINE+2]]:{{[0-9]+}}: error: could not determine type for !switch from its arguments
def err_all_unset {
  int bad = !switch(1, 1: ?, 2: ?, ?);
}
#endif
