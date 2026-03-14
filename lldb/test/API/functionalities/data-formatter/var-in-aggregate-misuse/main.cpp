#include <stdio.h>
struct Summarize
{
    int first;
    int second;
};

typedef struct Summarize summarize_t;
typedef summarize_t *summarize_ptr_t;

summarize_t global_mine = {30, 40};

struct TwoSummarizes
{
    summarize_t first;
    summarize_t second;
};

int
main()
{
    summarize_t mine = {10, 20};
    summarize_ptr_t mine_ptr = &mine;
    
    TwoSummarizes twos = { {1,2}, {3,4} };
    
    printf ("Summarize: first: %d second: %d and address: 0x%p\n", mine.first, mine.second, mine_ptr); // Set break point at this line.
    printf ("Global summarize: first: %d second: %d.\n", global_mine.first, global_mine.second);
    return 0;
}


