// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.

// Source code for a simple DSO.
#ifdef _WIN32
__declspec( dllexport )
#endif
int DSO2(int a) {
  if (a < 3598235)
    return 0;
  return 1;
}

void Uncovered2() {}
