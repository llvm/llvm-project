#include <thread>

#include "lldb/API/LLDB.h"
#include "lldb/API/SBDebugger.h"
#include "lldb/API/SBTarget.h"

using namespace lldb;
int main (int argc, char **argv)
{
  // We are expecting the program path and a path to an executable to load
  if (argc != 2)
    return 1;
  const char *program_file = argv[1];
  SBDebugger::Initialize();
  SBDebugger debugger = SBDebugger::Create(false);
  auto lambda = [&](){
    SBError error;
    SBTarget target = debugger.CreateTarget(program_file, nullptr, nullptr, 
                                            false, error);
  };

  // Create 3 targets at the same time and make sure we don't crash.
  std::thread thread1(lambda);
  std::thread thread2(lambda);
  std::thread thread3(lambda);
  thread1.join();
  thread2.join();
  thread3.join();
  SBDebugger::Terminate();
  return 0;
}
