#include <atomic>

int main()
{
    std::atomic<int> Q(1);
    return Q; // Set break point at this line.
}
