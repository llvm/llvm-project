#include <map>

int main()
{
    std::map<int, int> map;
    for (int i = 0;
    i < 1500;
    i++)
        map[i] = i;
    return map.size(); // break here
}
