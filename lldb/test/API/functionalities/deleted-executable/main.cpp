#include <chrono>
#include <fstream>
#include <thread>

int main() { int argc = 0; char **argv = (char **)0; 
    lldb_enable_attach();

    {
      // Create file to signal that this process has started up.
      std::ofstream f;
      f.open(argv[1]);
    }

    std::this_thread::sleep_for(std::chrono::seconds(60));
}
