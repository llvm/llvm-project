int globalVar = 0xDEADBEEF;
extern int externGlobalVar;

int *globalPtr = &globalVar;
int &globalRef = globalVar;

namespace ns {
int globalVar = 13;
int *globalPtr = &globalVar;
int &globalRef = globalVar;
} // namespace ns

int foo = 2;

int main(int argc, char **argv) {
  int foo = 1;
  return 0; // Set a breakpoint here
}
