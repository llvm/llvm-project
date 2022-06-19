// Evil hack: To simulate memory corruption, we want to fiddle with some internals of std::list.
// Make those accessible to us.
#define private public
#define protected public

#include <list>
#include <stdio.h>
#include <assert.h>

int main()
{
    std::list<int> numbers_list{1,2,3,4,5,6,7,8,9,10};
    printf("// Set break point at this line.");
    std::list<int>::iterator it1=numbers_list.begin();
    while (it1 != numbers_list.end()){
       *it1++;
    }
    *it1++;
    *it1++;
    *it1++;
    assert(*it1 == 3);
    *it1++;
    *it1++;
    assert(*it1 == 5);

    // Any attempt to free the list will probably crash the program. Let's just leak it.
    return 0; // Set second break point at this line.
}
