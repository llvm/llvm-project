#include <chrono>
#include <cstdlib>
#include <thread>

using namespace std;

int main(int argc, char *argv[]) {
  const char *foo = getenv("FOO");
  for (int counter = 1;; counter++) {
    this_thread::sleep_for(chrono::seconds(1)); // breakpoint
  }
  return 0;
}
