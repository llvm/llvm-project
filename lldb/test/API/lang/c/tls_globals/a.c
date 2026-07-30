#include <unistd.h>

__thread int var_shared = 33;

int
touch_shared()
{
    return var_shared;
}

void shared_check()
{
	var_shared *= 2;
	usleep(1); // shared thread breakpoint
}
