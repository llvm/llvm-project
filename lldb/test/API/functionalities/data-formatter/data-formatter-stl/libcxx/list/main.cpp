#include <string>
#include <list>
#include <stdio.h>

typedef std::list<int> int_list;
typedef std::list<std::string> string_list;

int main()
{
    int_list numbers_list;
    std::list<int>* list_ptr = &numbers_list;

    printf("// Set break point at this line.");
    (numbers_list.push_back(0x12345678));
    (numbers_list.push_back(0x11223344));
    (numbers_list.push_back(0xBEEFFEED));
    (numbers_list.push_back(0x00ABBA00));
    (numbers_list.push_back(0x0ABCDEF0));
    (numbers_list.push_back(0x0CAB0CAB));
    
    numbers_list.clear();
    
    (numbers_list.push_back(1));
    (numbers_list.push_back(2));
    (numbers_list.push_back(3));
    (numbers_list.push_back(4));
    
    string_list text_list;
    (text_list.push_back(std::string("goofy")));
    (text_list.push_back(std::string("is")));
    (text_list.push_back(std::string("smart")));
    
    printf("// Set second break point at this line.");
    (text_list.push_back(std::string("!!!"))); 
    
    std::list<int> countingList = {3141, 3142, 3142,3142,3142, 3142, 3142, 3141};
    countingList.sort();
    printf("// Set third break point at this line.");
    countingList.unique();
    printf("// Set fourth break point at this line.");
    countingList.size();

    return 0;
}
