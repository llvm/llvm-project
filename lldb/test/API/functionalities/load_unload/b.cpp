int b_init()
{
    return 345;
}

int b_global = b_init();

int LLDB_DYLIB_EXPORT b_function() { return 500; }
