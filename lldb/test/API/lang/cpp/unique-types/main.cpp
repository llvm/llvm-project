#include <vector>

#include <stdio.h>
#include <stdint.h>

int main() { int argc = 0; char **argv = (char **)0;

    std::vector<long> longs;
    std::vector<short> shorts;  
    for (int i=0; i<12; i++)
    {
        longs.push_back(i);
        shorts.push_back(i);
    }
    return 0; // Set breakpoint here to verify that std::vector 'longs' and 'shorts' have unique types.
}
