#include <string>
#ifdef _LIBCPP_INLINE_VISIBILITY
#undef _LIBCPP_INLINE_VISIBILITY
#endif
#define _LIBCPP_INLINE_VISIBILITY
#include <vector>
typedef std::vector<int> int_vect;
typedef std::vector<std::string> string_vect;

int main()
{
    int_vect numbers;
    (numbers.push_back(1));  // Set break point at this line.
    (numbers.push_back(12));
    (numbers.push_back(123));
    (numbers.push_back(1234));
    (numbers.push_back(12345));
    (numbers.push_back(123456));
    (numbers.push_back(1234567));
    
    numbers.clear();
    
    (numbers.push_back(7));

    string_vect strings;
    (strings.push_back(std::string("goofy")));
    (strings.push_back(std::string("is")));
    (strings.push_back(std::string("smart")));
    
    (strings.push_back(std::string("!!!"))); // Set second break point at this line.
    
    strings.clear(); 
    
    return 0;
}
