#include <cstdio>
#include <string>
#include <unistd.h>

std::string getline() {
  std::string result;
  while (true) {
    int r;
    char c;
    do
      r = read(fileno(stdin), &c, 1);
    while (r == -1 && errno == EINTR);
    if (r <= 0 || c == '\n')
      return result;
    result += c;
  }
}

void input_copy_loop() {
  std::string str;
  while (str = getline(), !str.empty())
    printf("read: %s\n", str.c_str());
}

int main() {
  input_copy_loop();
  return 0;
}
