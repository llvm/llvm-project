#include <string>
#include <map>

#define intint_map std::map<int, int> 
#define strint_map std::map<std::string, int> 
#define intstr_map std::map<int, std::string> 
#define strstr_map std::map<std::string, std::string> 

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
    
    ii[0] = 0; // Set break point at this line.
    ii[1] = 1;

    intint_map::iterator it = ii.begin();
    intint_map::const_iterator const_it = ii.cbegin();
    std::printf("%d %d\n", it->second, const_it->second);

    thefoo_rw(1); // Set break point at this line.
    ii[2] = 0;
    ii[3] = 1;
	thefoo_rw(1);  // Set break point at this line.
	ii[4] = 0;
    ii[5] = 1;
    ii[6] = 0;
    ii[7] = 1;
    thefoo_rw(1);  // Set break point at this line.
    ii[85] = 1234567;

    ii.clear();
    
    strint_map si;
    thefoo_rw(1);  // Set break point at this line.
	
    si["zero"] = 0;
	thefoo_rw(1);  // Set break point at this line.
    si["one"] = 1;
    si["two"] = 2;
    si["three"] = 3;
	thefoo_rw(1);  // Set break point at this line.
    si["four"] = 4;

    si.clear();
    thefoo_rw(1);  // Set break point at this line.
	
    intstr_map is;
    thefoo_rw(1);  // Set break point at this line.
    is[85] = "goofy";
    is[1] = "is";
    is[2] = "smart";
    is[3] = "!!!";
    thefoo_rw(1);  // Set break point at this line.
	
    is.clear();
    thefoo_rw(1);  // Set break point at this line.
	
    strstr_map ss;
    thefoo_rw(1);  // Set break point at this line.
	
    ss["ciao"] = "hello";
    ss["casa"] = "house";
    ss["gatto"] = "cat";
    thefoo_rw(1);  // Set break point at this line.
    ss["a Mac.."] = "..is always a Mac!";
    
    ss.clear();
    thefoo_rw(1);  // Set break point at this line.    
    return 0;
}
