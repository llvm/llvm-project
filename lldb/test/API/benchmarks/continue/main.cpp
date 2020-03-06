#include <map>

#define intint_map std::map<int, int> 

int g_the_foo = 0;

int thefoo_rw(int arg = 1)
{
	if (arg < 0)
		arg = 0;
	if (!arg)
		arg = 1;
	g_the_foo += arg;
	return g_the_foo;
}

int main()
{
    intint_map ii;
    
    for (int i = 0; i < 15; i++)
    {
        ii[i] = i + 1;
        thefoo_rw(i); // break here
    }

    ii.clear();

    for (int j = 0; j < 15; j++)
    {
        ii[j] = j + 1;
        thefoo_rw(j); // break here
    }
    
    return 0;
}
