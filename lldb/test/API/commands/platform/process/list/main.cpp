#include <chrono>
#include <fstream>
#include <thread>

int main(int argc, char const *argv[]) {
  std::ofstream(argv[1]).close();
  std::this_thread::sleep_for(std::chrono::seconds(30));
  return 0;
}
