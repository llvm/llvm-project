struct Empty {};
struct SingleMember {
  int i;
};

struct PaddingMember {
  int i;
  char c;
};

const unsigned sizeof_empty = sizeof(struct Empty);
const unsigned sizeof_single = sizeof(struct SingleMember);
const unsigned sizeof_padding = sizeof(struct PaddingMember);

int main() {
  struct Empty empty;
  struct SingleMember single;
  struct PaddingMember padding;
  // Make sure globals are used.
  return sizeof_empty + sizeof_single + sizeof_padding;
}
