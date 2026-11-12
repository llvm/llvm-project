struct Empty {};
class EmptyClass {};
class alignas(4) EmptyClassAligned {};
class ClassEmptyMember {
  int i[0];
};

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
const unsigned sizeof_empty_class_aligned = sizeof(EmptyClassAligned);
const unsigned sizeof_class_empty_member = sizeof(ClassEmptyMember);
const unsigned sizeof_single = sizeof(SingleMember);
const unsigned sizeof_single_class = sizeof(SingleMemberClass);
const unsigned sizeof_padding = sizeof(PaddingMember);
const unsigned sizeof_padding_class = sizeof(PaddingMemberClass);

int main() {
  Empty empty;
  EmptyClass empty_class;
  EmptyClassAligned empty_class_aligned;
  ClassEmptyMember class_empty_member;
  SingleMember single;
  SingleMemberClass single_class;
  PaddingMember padding;
  PaddingMemberClass padding_class;
  // Make sure globals are used.
  return sizeof_empty + sizeof_empty_class + sizeof_class_empty_member +
         sizeof_single + +sizeof_empty_class_aligned + sizeof_single_class +
         sizeof_padding + sizeof_padding_class;
}
