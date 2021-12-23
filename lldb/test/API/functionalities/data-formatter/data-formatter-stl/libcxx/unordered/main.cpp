#include <string>
#include <unordered_map>
#include <unordered_set>

using std::string;

#define intstr_map std::unordered_map<int, string> 
#define intstr_mmap std::unordered_multimap<int, string> 

#define int_set std::unordered_set<int> 
#define str_set std::unordered_set<string> 
#define int_mset std::unordered_multiset<int> 
#define str_mset std::unordered_multiset<string> 

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
	intstr_map map;
	map.emplace(1,"hello");
	map.emplace(2,"world");
	map.emplace(3,"this");
	map.emplace(4,"is");
	map.emplace(5,"me");
	thefoo_rw();  // Set break point at this line.
	
	intstr_mmap mmap;
	mmap.emplace(1,"hello");
	mmap.emplace(2,"hello");
	mmap.emplace(2,"world");
	mmap.emplace(3,"this");
	mmap.emplace(3,"this");
	mmap.emplace(3,"this");
	thefoo_rw();  // Set break point at this line.
	
	int_set iset;
	iset.emplace(1);
	iset.emplace(2);
	iset.emplace(3);
	iset.emplace(4);
	iset.emplace(5);
	thefoo_rw();  // Set break point at this line.
	
	str_set sset;
	sset.emplace("hello");
	sset.emplace("world");
	sset.emplace("this");
	sset.emplace("is");
	sset.emplace("me");
	thefoo_rw();  // Set break point at this line.
	
	int_mset imset;
	imset.emplace(1);
	imset.emplace(2);
	imset.emplace(2);
	imset.emplace(3);
	imset.emplace(3);
	imset.emplace(3);
	thefoo_rw();  // Set break point at this line.
	
	str_mset smset;
	smset.emplace("hello");
	smset.emplace("world");
	smset.emplace("world");
	smset.emplace("is");
	smset.emplace("is");
	thefoo_rw();  // Set break point at this line.
	
    return 0;
}
