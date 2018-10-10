// Build with "cl.exe /Z7 /GR- /GS- -EHs-c- every-function.cpp /link /debug /nodefaultlib /incremental:no /entry:main"

void __cdecl operator delete(void *, unsigned int) {}
void __cdecl operator delete(void *, unsigned __int64) {}

// Note: It's important that this particular function hashes to a higher bucket
// number than any other function in the PDB.  When changing this test, ensure
// that this requirement still holds.  This is because we need to test lookup
// in the edge case where we try to look something up in the final bucket, which
// has special logic.
int OvlGlobalFn(int X) { return X + 42; }
int OvlGlobalFn(int X, int Y) { return X + Y + 42; }
int OvlGlobalFn(int X, int Y, int Z) { return X + Y + Z + 42; }

static int StaticFn(int X) {
  return X + 42;
}

int main(int argc, char **argv) {
  // Make sure they don't get optimized out.
  int Result = OvlGlobalFn(argc) + OvlGlobalFn(argc, argc) + OvlGlobalFn(argc, argc, argc) + StaticFn(argc);
  return Result;
}
