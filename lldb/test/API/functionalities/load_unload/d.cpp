int d_init()
{
    return 123;
}

int d_global = d_init();

int
d_function ()
{ // Find this line number within d_dunction().
    return 700;
}
