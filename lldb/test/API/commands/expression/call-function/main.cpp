#include <cstring>
#include <string>

struct Five {
  int number;
  const char *name;
};

Five returnsFive() {
  Five my_five = {5, "five"};
  return my_five;
}

unsigned int fib(unsigned int n) {
  if (n < 2)
    return n;
  else
    return fib(n - 1) + fib(n - 2);
}

int add(int a, int b) { return a + b; }

bool stringCompare(const char *str) {
  if (strcmp(str, "Hello world") == 0)
    return true;
  else
    return false;
}

int main(int argc, char const *argv[]) {
  std::string str = "Hello world";
  Five main_five = returnsFive();
  return strlen(str.c_str()); // break here
}
