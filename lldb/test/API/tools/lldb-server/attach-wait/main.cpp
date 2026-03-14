#include <thread>
#include <fstream>

int main(int argc, char *argv[])
{
  if (argc >= 2) {
    std::ofstream(argv[1]).close();
  }
  // Wait to be attached.
  std::this_thread::sleep_for(std::chrono::minutes(1));
  return 0;
}
