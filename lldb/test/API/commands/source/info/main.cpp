int bar();

int foo() {
  return 3;
}

int main() {
  int f = foo() + bar();
  f++;
  return f; //%self.expect("source info", substrs=["Lines found in module ", "main.cpp:10"])
  //%self.expect("source info -f main.cpp -c 10", matching=True, substrs=["main.cpp:10"])
  //%self.expect("source info -f main.cpp -c 1", matching=False, substrs=["main.cpp:10"])
  //%self.expect("source info -f main.cpp -l 10", matching=False, substrs=["main.cpp:7"])
}
