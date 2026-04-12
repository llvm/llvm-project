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

int main() {
  S s{1, 2};
  return md1.get(s) + md2.get(s);
}
