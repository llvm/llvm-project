int step_out_of_here(int a) {
  return a + 5; // Set breakpoint here
}

int main() {
#line 0
  int v = step_out_of_here(3) + 7;
#line 9
  v += 11;  // Should stop here
  return v; // Ran too far
}
