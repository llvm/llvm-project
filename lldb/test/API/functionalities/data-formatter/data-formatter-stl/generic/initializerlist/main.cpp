#include <string>
#include <vector>
#include <initializer_list>

int main ()
{
    std::initializer_list<int> ili{1,2,3,4,5};
    std::initializer_list<std::string> ils{"1","2","3","4","surprise it is a long string!! yay!!"};
    
    return 0; // Set break point at this line.
}

