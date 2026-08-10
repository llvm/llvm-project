int bar ()
{
    return 5;
}
int foo ()
{
    return bar() + 5;
}
int main ()
{
    return foo();
}
