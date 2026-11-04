#ifdef _WIN32
#include <windows.h>
#define sleep(x) Sleep((x) * 1000)
#else
#include <unistd.h>
#endif

int main() {
  int count = 100;
  while (count--)
    sleep(1); // break here
}
