// Compile with "cl /c /Zi /GR- UsingNamespaceTest.cpp"
// Link with "link UsingNamespaceTest.obj /debug /nodefaultlib /entry:main"

namespace NS {
  int foo() { return 1; }
}

using namespace NS;
int main(int argc, char **argv) {
  return foo();
}
