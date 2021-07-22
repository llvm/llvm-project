struct Empty {};
class EmptyClass {};

struct SingleMember {
  int i;
};
class SingleMemberClass {
  int i;
};

struct PaddingMember {
  int i;
  char c;
};
class PaddingMemberClass {
  int i;
  char c;
};

const unsigned sizeof_empty = sizeof(Empty);
const unsigned sizeof_empty_class = sizeof(EmptyClass);
const unsigned sizeof_single = sizeof(SingleMember);
const unsigned sizeof_single_class = sizeof(SingleMemberClass);
const unsigned sizeof_padding = sizeof(PaddingMember);
const unsigned sizeof_padding_class = sizeof(PaddingMemberClass);

int main() {
  Empty empty;
  EmptyClass empty_class;
  SingleMember single;
  SingleMemberClass single_class;
  PaddingMember padding;
  PaddingMemberClass padding_class;
  // Make sure globals are used.
  return sizeof_empty + sizeof_empty_class + sizeof_single +
    sizeof_single_class + sizeof_padding + sizeof_padding_class;
}
