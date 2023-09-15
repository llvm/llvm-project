
#define BUFFER_SIZE 32
struct PointType {
  int x;
  int y;
  int buffer[BUFFER_SIZE];
};
#include <vector>
int g_global = 123;
static int s_global = 234;
int test_indexedVariables();
int main(int argc, char const *argv[]) {
  static float s_local = 2.25;
  PointType pt = { 11,22, {0}};
  PointType *pt_ptr = new PointType{11, 22, {0}};
  PointType *another_pt_ptr = nullptr;
  for (int i=0; i<BUFFER_SIZE; ++i)
    pt.buffer[i] = i;

  int some_int = 10;
  int *some_int_ptr = new int{20};
  int *another_int_ptr = nullptr;
  int x = s_global - g_global - pt.y; // breakpoint 1
  {
    int x = 42;
    {
      int x = 72;
      s_global = x; // breakpoint 2
    }
  }
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
