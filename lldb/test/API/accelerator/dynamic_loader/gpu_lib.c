// A trivial shared library the accelerator dynamic loader loads into the
// accelerator target. It only needs to be a real, parseable object file with
// some code; the functions are never called.
int gpu_lib_add(int a, int b) { return a + b; }

int gpu_lib_mul(int a, int b) { return a * b; }
