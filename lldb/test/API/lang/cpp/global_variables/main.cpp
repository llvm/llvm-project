#include <stdio.h>

namespace abc {
int g_file_global_int = 42;
const int g_file_global_const_int = 1337;

namespace {
const int g_anon_namespace_const_int = 100;
}
}

int main (int argc, char const *argv[])
{
  int unused = abc::g_file_global_const_int;
  int unused2 = abc::g_anon_namespace_const_int;
  return abc::g_file_global_int; // Set break point at this line.
}
