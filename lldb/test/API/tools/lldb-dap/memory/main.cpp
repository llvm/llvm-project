int main() {
  int not_a_ptr = 666;
  const char *rawptr = "dead";
  // Immutable variable, .rodata region.
  static const int nonWritable = 100;
  // Breakpoint
  return 0;
}
