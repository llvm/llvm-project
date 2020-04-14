int d_init()
{
    return 123;
}

int d_global = d_init();

int LLDB_DYLIB_EXPORT d_function() {
  return 700; // Find this line number within d_dunction().
}
