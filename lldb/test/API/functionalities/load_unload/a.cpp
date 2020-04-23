extern LLDB_DYLIB_IMPORT int b_function();

int a_init()
{
    return 234;
}

int a_global = a_init();

extern "C" int
a_function ()
{
    return b_function ();
}
