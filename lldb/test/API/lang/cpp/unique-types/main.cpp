#include <vector>

#include <stdio.h>
#include <stdint.h>

int main (int argc, char const *argv[], char const *envp[])
{
    std::vector<long> longs;
    std::vector<short> shorts;  
    for (int i=0; i<12; i++)
    {
        longs.push_back(i);
        shorts.push_back(i);
    }
    return 0; // Set breakpoint here to verify that std::vector 'longs' and 'shorts' have unique types.
}
