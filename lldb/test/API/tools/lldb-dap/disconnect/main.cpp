#include <chrono>
#include <cstdio>
#include <fstream>
#include <string>
#include <thread>

volatile bool wait_for_attach = true;

void handle_attach(char *sync_file_path) {
  lldb_enable_attach();

  {
    // Create a file to signal that this process has started up.
    std::ofstream sync_file;
    sync_file.open(sync_file_path);
  }

  while (wait_for_attach)
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
}

int main(int argc, char **args) {
  if (argc == 2)
    handle_attach(args[1]);

  // We let the binary live a little bit to see if it executed after detaching
  // from // breakpoint

  // Create a file to signal that this process has started up.
  std::ofstream out_file; // breakpoint
  out_file.open(std::string(args[0]) + ".side_effect");
  return 0;
}
