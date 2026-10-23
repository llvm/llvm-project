extern int return_bar();
extern int return_baz();

int main() {
  int bar = return_bar(); // Stop here
  bar += return_bar();
  int baz = return_baz(); // Run to here
  baz += return_baz();

  return bar + baz;
}
