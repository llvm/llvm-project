void test()
{
    typedef double * a;
    a b = 0; // Set break point 4.
}
int main (int argc, char const *argv[])
{
    typedef float a;
    int i = 0; // Set break point 1.
    i++;
    a floatvariable = 2.7; // Set break point 2.
    {
        typedef char a;
        i++;
        a charvariable = 'a'; // Set break point 3.
        test();
    }
    {
        int c = 0;
        c++; // Set break point 5.
        for(i = 0 ; i < 1 ; i++)
        {
            typedef int a;
            a b;
            b = 7; // Set break point 6.
        }
        for(i = 0 ; i < 1 ; i++)
        {
            typedef double a;
            a b;
            b = 3.14; // Set break point 7.
        }
        c = 1; // Set break point 8.
    }
    floatvariable = 2.5;
    floatvariable = 2.8; // Set break point 9.
    return 0;
}
