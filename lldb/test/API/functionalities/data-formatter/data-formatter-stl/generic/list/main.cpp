#include <list>
#include <string>

typedef std::list<int> int_list;
typedef std::list<std::string> string_list;


template <typename T> void by_ref_and_ptr(T &ref, T *ptr) {
  // Check ref and ptr
  return;
}

int main()
{
    int_list numbers_list;
    
    numbers_list.push_back(0x12345678); // Set break point at this line.
    numbers_list.push_back(0x11223344);
    numbers_list.push_back(0xBEEFFEED);
    numbers_list.push_back(0x00ABBA00);
    numbers_list.push_back(0x0ABCDEF0);
    numbers_list.push_back(0x0CAB0CAB);
    
    numbers_list.clear();
    
    numbers_list.push_back(1);
    numbers_list.push_back(2);
    numbers_list.push_back(3);
    numbers_list.push_back(4);

    by_ref_and_ptr(numbers_list, &numbers_list);
    
    string_list text_list;
    text_list.push_back(std::string("goofy")); // Optional break point at this line.
    text_list.push_back(std::string("is"));
    text_list.push_back(std::string("smart"));
    text_list.push_back(std::string("!!!"));

    by_ref_and_ptr(text_list, &text_list);
        
    return 0; // Set final break point at this line.
}

