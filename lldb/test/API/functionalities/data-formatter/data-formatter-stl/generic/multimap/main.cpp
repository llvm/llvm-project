#include <string>
#include <map>

#define intint_map std::multimap<int, int> 
#define strint_map std::multimap<std::string, int> 
#define intstr_map std::multimap<int, std::string> 
#define strstr_map std::multimap<std::string, std::string> 

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
    
    ii.emplace(0,0); // Set break point at this line.
    ii.emplace(1,1);
	thefoo_rw(1);  // Set break point at this line.
    ii.emplace(2,0);
	ii.emplace(3,1);
	thefoo_rw(1);  // Set break point at this line.
	ii.emplace(4,0);
	ii.emplace(5,1);
	ii.emplace(6,0);
	ii.emplace(7,1);
    thefoo_rw(1);  // Set break point at this line.
    ii.emplace(85,1234567);

    ii.clear();
    
    strint_map si;
    thefoo_rw(1);  // Set break point at this line.
	
    si.emplace("zero",0);
	thefoo_rw(1);  // Set break point at this line.
	si.emplace("one",1);
	si.emplace("two",2);
	si.emplace("three",3);
	thefoo_rw(1);  // Set break point at this line.
	si.emplace("four",4);

    si.clear();
    thefoo_rw(1);  // Set break point at this line.
	
    intstr_map is;
    thefoo_rw(1);  // Set break point at this line.
    is.emplace(85,"goofy");
    is.emplace(1,"is");
    is.emplace(2,"smart");
    is.emplace(3,"!!!");
    thefoo_rw(1);  // Set break point at this line.
	
    is.clear();
    thefoo_rw(1);  // Set break point at this line.
	
    strstr_map ss;
    thefoo_rw(1);  // Set break point at this line.
	
    ss.emplace("ciao","hello");
    ss.emplace("casa","house");
    ss.emplace("gatto","cat");
    thefoo_rw(1);  // Set break point at this line.
    ss.emplace("a Mac..","..is always a Mac!");
    
    ss.clear();
    thefoo_rw(1);  // Set break point at this line.    
    return 0;
}
