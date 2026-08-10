int bar() {
  int i = 3; //%self.expect("source info", substrs=["Lines found in module ", "second.cpp:2"])
  return i; //%self.expect("source info", substrs=["Lines found in module ", "second.cpp:3"])
  //%self.expect("source info --name main", substrs=["Lines found in module ", "main.cpp:7", "main.cpp:10"])
}
