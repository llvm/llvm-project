
#define BUFFER_SIZE 16
struct PointType {
  int x;
  int y;
  int buffer[BUFFER_SIZE];
};
#include <cstdio>
#include <vector>
extern int g_global;
int g_global = 123;
static int s_global = 234;
int test_indexedVariables();
int test_return_variable();
int test_anonymous_types();
int test_anonymous_fields();
void test_unnamed_bitfields();

int main(int argc, char const *argv[]) {
  static float s_local = 2.25;
  PointType pt = {11, 22, {0}};
  for (int i = 0; i < BUFFER_SIZE; ++i)
    pt.buffer[i] = i;
  const char *valid_str = "𐌶𐌰L𐌾𐍈 C𐍈𐌼𐌴𐍃";
  const char *malformed_str = "lone trailing \x81\x82 bytes";
  printf("print malformed utf8 %s %s\n", valid_str, malformed_str);
  int x = s_global - g_global - pt.y; // breakpoint 1
  {
    int x = 42;
    {
      int x = 72;
      s_global = x; // breakpoint 2
    }
  }
  {
    int return_result = test_return_variable();
  }
  test_anonymous_types();
  test_anonymous_fields();
  test_unnamed_bitfields();
  return test_indexedVariables(); // breakpoint 3
}

int test_indexedVariables() {
  int small_array[5] = {1, 2, 3, 4, 5};
  int large_array[200];
  std::vector<int> small_vector;
  std::vector<int> large_vector;
  small_vector.assign(5, 0);
  large_vector.assign(200, 0);
  return 0; // breakpoint 4
}

int test_return_variable() {
  return 300; // breakpoint 5
}

int test_anonymous_types() {
  struct {
    char name[16];
    int x, y;
  } my_var = {"hello world!", 42, 7};
  return my_var.x + my_var.y; // breakpoint 6
}

struct MySock {
  union {
    unsigned char ipv4[4];
    unsigned char ipv6[6];
  };
};

int test_anonymous_fields() {
  MySock home = {{{0}}};
  home.ipv4[0] = 127;
  home.ipv4[1] = 0;
  home.ipv4[2] = 0;
  home.ipv4[1] = 1;
  return 1; // breakpoint 7
}

void test_unnamed_bitfields() {
  struct example {
    unsigned int lo : 4;
    unsigned int : 0;
    unsigned int hi : 4;
  };
  example e = {0xA, 0xB};
  printf("lo: %u, hi: %u\n", e.lo, e.hi); // breakpoint 8
}
