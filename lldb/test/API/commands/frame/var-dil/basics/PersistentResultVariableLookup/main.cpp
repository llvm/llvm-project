typedef struct {
  char charm;
} NestedT;

typedef struct {
  int intm;
  double doublem;
  NestedT nestedm;

} HasMembersT;

int main(int argc, char **argv) {
  HasMembersT hsmt;

  hsmt.nestedm.charm = 'c';
  hsmt.intm = 1;
  hsmt.doublem = 2.0;

  int foo = 1;
  foo = 2; // Set a breakpoint here
  hsmt.doublem = 3.0;
  return 0; // Set a second breakpoint here
}
