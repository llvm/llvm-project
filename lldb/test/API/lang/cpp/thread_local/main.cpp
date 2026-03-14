int storage = 45;
thread_local int tl_global_int = 123;
thread_local int *tl_global_ptr = &storage;

int main(int argc, char **argv) {
  thread_local int tl_local_int = 321;
  thread_local int *tl_local_ptr = nullptr;
  tl_local_ptr = &tl_local_int;
  tl_local_int++;
  return 0; // Set breakpoint here
}
