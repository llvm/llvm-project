int storage = 45;
thread_local int tl_global_int = 123;
thread_local int *tl_global_ptr = &storage;

int main(int argc, char **argv) {
  //% self.expect("expr tl_local_int", error=True, substrs=["couldn't get the value of variable tl_local_int"])
  //% self.expect("expr *tl_local_ptr", error=True, substrs=["couldn't get the value of variable tl_local_ptr"])
  thread_local int tl_local_int = 321;
  thread_local int *tl_local_ptr = nullptr;
  tl_local_ptr = &tl_local_int;
  tl_local_int++;
  //% self.expect("expr tl_local_int + 1", substrs=["int", "= 323"])
  //% self.expect("expr *tl_local_ptr + 2", substrs=["int", "= 324"])
  //% self.expect("expr tl_global_int", substrs=["int", "= 123"])
  //% self.expect("expr *tl_global_ptr", substrs=["int", "= 45"])
  return 0;
}
