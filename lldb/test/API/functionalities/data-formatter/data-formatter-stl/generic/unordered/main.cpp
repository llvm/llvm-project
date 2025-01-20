#include <string>
#include <unordered_map>
#include <unordered_set>

int g_the_foo = 0;

int thefoo_rw(int arg = 1) {
  if (arg < 0)
    arg = 0;
  if (!arg)
    arg = 1;
  g_the_foo += arg;
  return g_the_foo;
}

int main() {

  char buffer[sizeof(std::unordered_map<int, std::string>)] = {0};
  std::unordered_map<int, std::string> &corrupt_map = *(std::unordered_map<int, std::string> *)buffer;

  // Make a typedef to ensure functionality when typedefs are used.
  typedef std::unordered_map<int, std::string> UnorderedMap;
  UnorderedMap map; // Set break point at this line.
  map.emplace(1, "hello");
  map.emplace(2, "world");
  map.emplace(3, "this");
  map.emplace(4, "is");
  map.emplace(5, "me");
  thefoo_rw(); // Set break point at this line.

  // Make a typedef to ensure functionality when typedefs are used.
  typedef std::unordered_multimap<int, std::string> UnorderedMultiMap;
  UnorderedMultiMap mmap;
  mmap.emplace(1, "hello");
  mmap.emplace(2, "hello");
  mmap.emplace(2, "world");
  mmap.emplace(3, "this");
  mmap.emplace(3, "this");
  mmap.emplace(3, "this");
  thefoo_rw(); // Set break point at this line.

  // Make a typedef to ensure functionality when typedefs are used.
  typedef std::unordered_set<int> IntsUnorderedSet;
  IntsUnorderedSet iset;
  iset.emplace(1);
  iset.emplace(2);
  iset.emplace(3);
  iset.emplace(4);
  iset.emplace(5);
  thefoo_rw(); // Set break point at this line.

  // Make a typedef to ensure functionality when typedefs are used.
  typedef std::unordered_set<std::string> StringsUnorderedSet;
  StringsUnorderedSet sset;
  sset.emplace("hello");
  sset.emplace("world");
  sset.emplace("this");
  sset.emplace("is");
  sset.emplace("me");
  thefoo_rw(); // Set break point at this line.

  // Make a typedef to ensure functionality when typedefs are used.
  typedef std::unordered_multiset<int> IntsUnorderedMultiSet;
  IntsUnorderedMultiSet imset;
  imset.emplace(1);
  imset.emplace(2);
  imset.emplace(2);
  imset.emplace(3);
  imset.emplace(3);
  imset.emplace(3);
  thefoo_rw(); // Set break point at this line.

  // Make a typedef to ensure functionality when typedefs are used.
  typedef std::unordered_multiset<std::string> StringsUnorderedMultiSet;
  StringsUnorderedMultiSet smset;
  smset.emplace("hello");
  smset.emplace("world");
  smset.emplace("world");
  smset.emplace("is");
  smset.emplace("is");
  thefoo_rw(); // Set break point at this line.

  return 0;
}
