
/// LLDB C API Test Driver

#include <algorithm>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

#include "lldb-headers.h"

#include "common.h"

using namespace std;
using namespace lldb;

void test(SBDebugger &dbg, std::vector<string> args);

int main(int argc, char** argv) {
  int code = 0;

  SBDebugger::Initialize();
  SBDebugger dbg = SBDebugger::Create();

  try {
    if (!dbg.IsValid())
      throw Exception("invalid debugger");
    vector<string> args(argv + 1, argv + argc);

    test(dbg, args);
  } catch (Exception &e) {
    cout << "ERROR: " << e.what() << endl;
    code = 1;
  }

  SBDebugger::Destroy(dbg);
  return code;
}
