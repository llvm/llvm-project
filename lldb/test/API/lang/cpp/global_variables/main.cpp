#include <stdio.h>

namespace abc {
	int g_file_global_int = 42;
}

int main (int argc, char const *argv[])
{
    return abc::g_file_global_int; // Set break point at this line.
}
