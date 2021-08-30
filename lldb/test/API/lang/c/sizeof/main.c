struct Empty {};
struct EmptyMember {
  char i[0];
};
struct SingleMember {
  int i;
};

struct PaddingMember {
  int i;
  char c;
};

const unsigned sizeof_empty = sizeof(struct Empty);
const unsigned sizeof_empty_member = sizeof(struct EmptyMember);
const unsigned sizeof_single = sizeof(struct SingleMember);
const unsigned sizeof_padding = sizeof(struct PaddingMember);

int main() {
  struct Empty empty;
  struct EmptyMember empty_member;
  struct SingleMember single;
  struct PaddingMember padding;
  // Make sure globals are used.
  return sizeof_empty + sizeof_empty_member + sizeof_single + sizeof_padding;
}
