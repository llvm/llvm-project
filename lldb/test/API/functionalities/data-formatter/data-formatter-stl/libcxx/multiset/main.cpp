#include <string>
#include <set>

typedef std::multiset<int> intset;
typedef std::multiset<std::string> stringset;

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

void by_ref_and_ptr(intset &ref, intset *ptr)
{
    // Stop here to check by ref and ptr
    return;
} 

int main()
{
    intset ii;
    thefoo_rw(1);  // Set break point at this line.
	
	ii.insert(0);
	ii.insert(1);
	ii.insert(2);
	ii.insert(3);
	ii.insert(4);
	ii.insert(5);
    thefoo_rw(1);  // Set break point at this line.

	ii.insert(6);
	thefoo_rw(1);  // Set break point at this line.

        by_ref_and_ptr(ii, &ii);

	ii.clear();
	thefoo_rw(1);  // Set break point at this line.

	stringset ss;
	thefoo_rw(1);  // Set break point at this line.

	ss.insert("a");
	ss.insert("a very long string is right here");
	thefoo_rw(1);  // Set break point at this line.

	ss.insert("b");
	ss.insert("c");
	thefoo_rw(1);  // Set break point at this line.
	
	ss.erase("b");
	thefoo_rw(1);  // Set break point at this line.

    return 0;
}
