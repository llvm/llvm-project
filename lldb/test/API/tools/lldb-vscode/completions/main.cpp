#include <string>
#include <vector>

int fun(std::vector<std::string> var) {
  return var.size(); // breakpoint 1
}

int main(int argc, char const *argv[]) {
  int var1 = 0;
  int var2 = 1;
  std::string str1 = "a";
  std::string str2 = "b";
  std::vector<std::string> vec;
  fun(vec);
  return 0; // breakpoint 2
}
