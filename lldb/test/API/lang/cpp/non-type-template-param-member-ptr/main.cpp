// Test that LLDB correctly handles non-type template parameters (NTTPs)
// of various kinds beyond plain integers.
//
// DWARF encodes each NTTP as a DW_TAG_template_value_parameter with a
// DW_AT_const_value. MakeAPValue must convert these values into clang
// APValues so that each specialization gets a distinct TemplateArgument.
//
// Without the fix, MakeAPValue only handled integer/enum/float types.
// Member data pointer and nullptr NTTPs were silently dropped, causing
// duplicate-specialization bugs.

struct S {
  int x;
  int y;
};

// --- Member data pointer NTTP ---
template <int S::*P> struct MemberData {
  int get(S &s) { return s.*P; }
};
MemberData<&S::x> md1;
MemberData<&S::y> md2;

// --- nullptr NTTP ---
int g1 = 10;
template <int *P> struct MaybeNull {
  bool is_null() { return P == nullptr; }
};
MaybeNull<nullptr> mn1;
MaybeNull<&g1> mn2;

int main() {
  S s{1, 2};
  return md1.get(s) + md2.get(s);
}
