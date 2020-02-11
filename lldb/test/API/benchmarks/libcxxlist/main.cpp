#include <list>

int main()
{
    std::list<int> list;
    for (int i = 0;
    i < 1500;
    i++)
        list.push_back(i);
    return list.size(); // break here
}
