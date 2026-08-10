void recursive_function(int i)
{
    if (i < 10)
    {
        recursive_function(i + 1);
    }
    else
    {
        char *null=0;
        *null = 0; // Crash here.
    }
}

int main(int argc, char *argv[])
{
    recursive_function(0);
    return 0;
}

