struct EmptyStruct {};

struct SingleChildStruct {
  int x;
};

struct TwoChildrenStruct {
  int x;
  int y;
};

int main() {
  EmptyStruct e;
  EmptyStruct &e_ref = e;

  SingleChildStruct s{0};
  SingleChildStruct &s_ref = s;

  TwoChildrenStruct t{0};
  TwoChildrenStruct &t_ref = t;

  return 0; // break here
}
